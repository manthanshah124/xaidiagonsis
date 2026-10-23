from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        """Load the ML model and OOD detector when Django starts."""
        try:
            from .services.model_loader import get_model_loader
            from .services.ood_detector import OODDetector
            from django.conf import settings
            from pathlib import Path

            loader = get_model_loader()
            model_path = settings.BASE_DIR / 'ml' / 'models' / 'plant_model.keras'
            class_names_path = settings.BASE_DIR / 'ml' / 'models' / 'class_names.json'
            gradcam_layer = getattr(settings, 'GRADCAM_LAYER_NAME', 'Conv_1')

            if model_path.exists() and class_names_path.exists():
                logger.info(f"Loading model from {model_path}")
                loader.load_model(str(model_path), gradcam_layer)
                loader.load_class_names(str(class_names_path))
                
                # Initialize OOD detector
                model = loader.get_model()
                class_names = loader.get_class_names()
                if model is not None and class_names is not None:
                    ood_detector = OODDetector(model, class_names)
                    loader.set_ood_detector(ood_detector)
                
                logger.info("Model, class names, and OOD detector loaded successfully")
            else:
                logger.warning(f"Model files not found. Model: {model_path.exists()}, Classes: {class_names_path.exists()}")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")