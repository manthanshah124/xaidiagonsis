"""Views for XAI-PlantDoc predictor app."""

import base64
import io
import json
import logging
from typing import Optional

import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image

from .services.model_loader import get_model_loader
from .services.preprocessing import (
    preprocess_for_model,
    PreprocessingError,
    MAX_FILE_SIZE,
)
from .services.ood_detector import OODDetector
from .services.predict import predict_disease, format_disease_name
from .services.explain_gradcam import generate_gradcam_overlay_base64

logger = logging.getLogger(__name__)


def index(request):
    """Serve the main single-page application."""
    return render(request, 'index.html')


@require_http_methods(["POST"])
def predict(request):
    """
    Handle image upload, validate, preprocess, predict, and generate Grad-CAM.
    Returns JSON with prediction results and Grad-CAM overlay.
    """
    # Get uploaded file
    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        return JsonResponse({
            'status': 'error',
            'code': 'NO_FILE',
            'message': 'No image file provided. Use form field "image".'
        }, status=400)

    # Read file bytes
    try:
        image_bytes = uploaded_file.read()
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'code': 'READ_ERROR',
            'message': f'Failed to read uploaded file: {str(e)}'
        }, status=400)

    # Validate and preprocess
    try:
        model_input, resized_rgb = preprocess_for_model(image_bytes, uploaded_file.name)
    except PreprocessingError as e:
        status_code = 413 if e.code == 'TOO_LARGE' else 400
        return JsonResponse({
            'status': 'error',
            'code': e.code,
            'message': e.message
        }, status=status_code)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return JsonResponse({
            'status': 'error',
            'code': 'PREPROCESSING_ERROR',
            'message': f'Preprocessing failed: {str(e)}'
        }, status=500)

    # Load model
    loader = get_model_loader()
    model = loader.get_model()
    class_names = loader.get_class_names()
    gradcam_layer = loader.get_gradcam_layer_name()

    if model is None or class_names is None:
        return JsonResponse({
            'status': 'error',
            'code': 'MODEL_NOT_LOADED',
            'message': 'Model not loaded. Please check server configuration.'
        }, status=503)

    # --- OOD Detection ---
    ood_detector = loader.get_ood_detector()
    if ood_detector:
        try:
            ood_result = ood_detector.predict_with_ood(model_input, resized_rgb)
            
            if ood_result['is_ood']:
                return JsonResponse({
                    'status': 'rejected',
                    'reason': ood_result['reason'],
                    'message': ood_result['message'],
                    'max_confidence': ood_result['max_confidence'],
                    'entropy': ood_result['entropy'],
                    'norm_entropy': ood_result.get('norm_entropy')
                })
        except Exception as e:
            logger.warning(f"OOD detection failed, proceeding with normal prediction: {e}")
            # Continue with normal prediction if OOD fails

    # --- Normal Prediction ---
    try:
        predicted_class, confidence, top3 = predict_disease(model, model_input, class_names)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JsonResponse({
            'status': 'error',
            'code': 'PREDICTION_ERROR',
            'message': f'Prediction failed: {str(e)}'
        }, status=500)

    # Generate Grad-CAM overlay
    gradcam_base64 = None
    try:
        class_idx = class_names.index(predicted_class) if predicted_class in class_names else 0
        original_image = resized_rgb.astype(np.uint8)
        gradcam_base64 = generate_gradcam_overlay_base64(
            model=model,
            preprocessed_image=model_input,
            original_image=original_image,
            class_idx=class_idx,
            layer_name=loader.get_gradcam_layer_name() or 'Conv_1',
            alpha=getattr(settings, 'GRADCAM_ALPHA', 0.4),
            colormap=cv2.COLORMAP_JET
        )
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    # Convert original image to base64
    try:
        original_pil = Image.fromarray(resized_rgb.astype(np.uint8))
        original_buffer = io.BytesIO()
        original_pil.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to encode original image: {e}")
        original_base64 = None

    # Warnings
    warnings = []
    low_conf_threshold = getattr(settings, 'LOW_CONFIDENCE_THRESHOLD', 0.6)
    if confidence < low_conf_threshold:
        warnings.append(
            f"Low confidence prediction ({confidence:.1%}). "
            f"Consider retaking the photo or consulting an expert."
        )

    display_name = format_disease_name(predicted_class)

    return JsonResponse({
        'status': 'ok',
        'disease': predicted_class,
        'display_name': display_name,
        'confidence': confidence,
        'top3': [{'class': c, 'confidence': conf} for c, conf in top3],
        'original_image_base64': original_base64,
        'gradcam_image_base64': gradcam_base64,
        'warnings': warnings,
        'message': 'Prediction completed successfully.'
    })


@require_http_methods(["GET"])
def health(request):
    """Health check endpoint."""
    loader = get_model_loader()
    return JsonResponse({
        'status': 'ok',
        'model_loaded': loader.is_loaded()
    })


@require_http_methods(["GET"])
def csrf_token_view(request):
    """Return CSRF token for AJAX requests."""
    return JsonResponse({'csrfToken': get_token(request)})