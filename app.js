/* XAI-PlantDoc frontend with auth, dashboard, and Grad-CAM */
(() => {
  "use strict";

  // Backend API endpoint (Django)
  const API_URL = "/predict/";
  const MAX_FILE_BYTES = 5 * 1024 * 1024;
  const ALLOWED_TYPES = ["image/jpeg", "image/png"];
  const LOW_CONFIDENCE_THRESHOLD = 60; // percent

  // --- DOM elements ---
  // Auth
  const authScreen = document.getElementById("auth-screen");
  const appScreen = document.getElementById("app-screen");
  const signinTab = document.getElementById("signin-tab");
  const signupTab = document.getElementById("signup-tab");
  const signinForm = document.getElementById("signin-form");
  const signupForm = document.getElementById("signup-form");
  const authMessage = document.getElementById("auth-message");
  const signoutButton = document.getElementById("signout-btn");
  const userDisplayName = document.getElementById("user-display-name");
  const userAvatar = document.getElementById("user-avatar");

  // App tabs
  const appTabs = document.querySelectorAll(".app-tab");
  const tabPanels = {
    "overview": document.getElementById("overview-panel"),
    "crop-disease": document.getElementById("crop-disease-panel"),
    "dashboard": document.getElementById("dashboard-panel"),
  };

  // Overview quick action
  const overviewAnalyzeBtn = document.getElementById("overview-analyze-btn");

  // Upload / prediction
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const previewFilename = document.getElementById("preview-filename");
  const removeButton = document.getElementById("remove-image");
  const predictButton = document.getElementById("predict-btn");
  const predictButtonLabel = document.getElementById("predict-btn-label");
  const errorBanner = document.getElementById("error-banner");
  const errorMessage = document.getElementById("error-message");
  const dismissError = document.getElementById("dismiss-error");
  const successBanner = document.getElementById("success-banner");
  const successMessage = document.getElementById("success-message");
  const resultEmpty = document.getElementById("result-empty");
  const resultContent = document.getElementById("result-content");
  const loadingState = document.getElementById("loading-state");
  const resultIndicator = document.getElementById("result-indicator");
  const predictionDisease = document.getElementById("prediction-disease");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceFill = document.getElementById("confidence-fill");
  const predictionMessage = document.getElementById("prediction-message");

  // Grad-CAM
  const gradcamSection = document.getElementById("gradcam-section");
  const gradcamOriginal = document.getElementById("gradcam-original");
  const gradcamOverlay = document.getElementById("gradcam-overlay");

  // Top-3
  const top3Section = document.getElementById("top3-section");
  const top3List = document.getElementById("top3-list");

  // Low confidence warning
  const lowConfidenceWarning = document.getElementById("low-confidence-warning");
  const lowConfidenceMessage = document.getElementById("low-confidence-message");

  // --- State ---
  let selectedFile = null;
  let previewUrl = null;
  let currentUser = null;

  // --- Helpers ---
  function showAuthMessage(message, type = "error") {
    authMessage.textContent = message;
    authMessage.className = "auth-message " + type;
    authMessage.hidden = false;
  }

  function hideAuthMessage() {
    authMessage.hidden = true;
    authMessage.className = "auth-message";
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorBanner.hidden = false;
    successBanner.hidden = true;
  }

  function showSuccess(message) {
    successMessage.textContent = message;
    successBanner.hidden = false;
    errorBanner.hidden = true;
  }

  function clearMessages() {
    errorBanner.hidden = true;
    successBanner.hidden = true;
  }

  function clearPreviewUrl() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  }

  function resetResult() {
    resultEmpty.hidden = false;
    resultContent.hidden = true;
    loadingState.hidden = true;
    gradcamSection.hidden = true;
    top3Section.hidden = true;
    lowConfidenceWarning.hidden = true;
    predictionDisease.textContent = "—";
    confidenceValue.textContent = "—";
    confidenceFill.style.width = "0%";
    predictionMessage.textContent = "";
    resultIndicator.textContent = "Ready";
    resultIndicator.className = "result-indicator";
  }

  // --- Auth (localStorage demo) ---
  function getUsers() {
    return JSON.parse(localStorage.getItem("xai_plantdoc_users") || "[]");
  }

  function saveUsers(users) {
    localStorage.setItem("xai_plantdoc_users", JSON.stringify(users));
  }

  function getCurrentUser() {
    return JSON.parse(localStorage.getItem("xai_plantdoc_current") || "null");
  }

  function setCurrentUser(user) {
    currentUser = user;
    if (user) {
      localStorage.setItem("xai_plantdoc_current", JSON.stringify(user));
      userDisplayName.textContent = user.name;
      userAvatar.textContent = user.name.charAt(0).toUpperCase();
      showApp();
    } else {
      localStorage.removeItem("xai_plantdoc_current");
      showAuth();
    }
  }

  function showAuth() {
    authScreen.hidden = false;
    appScreen.hidden = true;
  }

  function showApp() {
    authScreen.hidden = true;
    appScreen.hidden = false;
    switchTab("overview");
  }

  function switchAuthTab(tab) {
    if (tab === "signup") {
      signinTab.setAttribute("aria-selected", "false");
      signupTab.setAttribute("aria-selected", "true");
      signinForm.hidden = true;
      signupForm.hidden = false;
    } else {
      signinTab.setAttribute("aria-selected", "true");
      signupTab.setAttribute("aria-selected", "false");
      signinForm.hidden = false;
      signupForm.hidden = true;
    }
    hideAuthMessage();
  }

  function switchTab(tabName) {
    appTabs.forEach((tab) => {
      const isActive = tab.dataset.tab === tabName;
      tab.setAttribute("aria-selected", isActive);
    });
    Object.entries(tabPanels).forEach(([key, panel]) => {
      panel.hidden = key !== tabName;
    });
  }

  // --- File handling ---
  function setFile(file) {
    clearMessages();
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      showError("Please select a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      showError("The image is larger than 5 MB. Please choose a smaller file.");
      return;
    }

    clearPreviewUrl();
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    previewFilename.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    previewContainer.hidden = false;
    dropZone.hidden = true;
    predictButton.disabled = false;
    resetResult();
    showSuccess("Image selected. Click “Analyze disease” to get a prediction.");
  }

  function clearSelectedFile() {
    clearPreviewUrl();
    selectedFile = null;
    fileInput.value = "";
    imagePreview.removeAttribute("src");
    previewFilename.textContent = "";
    previewContainer.hidden = true;
    dropZone.hidden = false;
    predictButton.disabled = true;
    clearMessages();
    resetResult();
  }

  // --- CSRF token for Django ---
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + "=")) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // --- Prediction ---
  async function predictDisease(file) {
    const formData = new FormData();
    formData.append("image", file);

    // Get CSRF token
    const csrftoken = getCookie("csrftoken");

    predictButton.disabled = true;
    predictButtonLabel.textContent = "Analyzing image…";
    resultEmpty.hidden = true;
    resultContent.hidden = true;
    loadingState.hidden = false;
    resultIndicator.textContent = "Analyzing…";
    resultIndicator.className = "result-indicator processing";

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "X-CSRFToken": csrftoken,
        },
        body: formData,
      });

      let data = {};
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const responseText = await response.text();
        throw new Error(responseText || `Server returned HTTP ${response.status}`);
      }

      if (!response.ok) {
        throw new Error(data.error || data.message || `Request failed with HTTP ${response.status}`);
      }

      // Handle OOD/rejected responses
      if (data.status === 'rejected') {
        loadingState.hidden = true;
        resultEmpty.hidden = false;
        resultContent.hidden = true;
        resultIndicator.textContent = "Rejected";
        resultIndicator.className = "result-indicator";
        
        let friendlyMsg = '';
        if (data.reason === 'not_leaf') {
          friendlyMsg = `🍃 ${data.message} Please upload a clear leaf photo.`;
        } else if (data.reason === 'low_confidence') {
          friendlyMsg = `❓ ${data.message} This disease may not be in our trained classes (14 tomato/apple diseases).`;
        } else if (data.reason === 'high_entropy') {
          friendlyMsg = `🤔 ${data.message} Try a clearer, well-lit leaf photo.`;
        } else {
          friendlyMsg = data.message;
        }
        
        showError(friendlyMsg);
        predictButton.disabled = !selectedFile;
        predictButtonLabel.textContent = "Analyze disease";
        return;
      }

      // Parse Django API response (normal ok response)
      // Expected format:
      // {
      //   "status": "ok",
      //   "disease": "Tomato_Early_blight",
      //   "display_name": "Tomato Early blight",
      //   "confidence": 0.94,
      //   "top3": [{"class": "...", "confidence": 0.94}, ...],
      //   "original_image_base64": "base64...",
      //   "gradcam_image_base64": "base64...",
      //   "warnings": [...],
      //   "message": "..."
      // }
      const label = data.disease || data.predicted_class || data.prediction || data.class_name;
      const displayName = data.display_name;
      const rawConfidence = data.confidence;

      if (!label) {
        throw new Error("The API responded, but no prediction label was found.");
      }

      const confidence = rawConfidence === undefined || rawConfidence === null
        ? null
        : (Number(rawConfidence) <= 1 ? Number(rawConfidence) * 100 : Number(rawConfidence));

      // Display main prediction
      predictionDisease.textContent = displayName || String(label).replaceAll("_", " ");
      confidenceValue.textContent = Number.isFinite(confidence) ? `${confidence.toFixed(1)}%` : "Not provided";
      confidenceFill.style.width = Number.isFinite(confidence)
        ? `${Math.max(0, Math.min(100, confidence))}%`
        : "0%";
      predictionMessage.textContent = data.message || "Prediction received from the backend API.";

      // Low confidence warning (from backend warnings or threshold)
      const hasLowConfidenceWarning = data.warnings && data.warnings.length > 0;
      if (hasLowConfidenceWarning) {
        lowConfidenceMessage.textContent = data.warnings[0];
        lowConfidenceWarning.hidden = false;
      } else if (Number.isFinite(confidence) && confidence < LOW_CONFIDENCE_THRESHOLD) {
        lowConfidenceMessage.textContent = `Confidence is low (${confidence.toFixed(1)}%). The model may be uncertain. Consider retaking the photo with better lighting or a clearer leaf view.`;
        lowConfidenceWarning.hidden = false;
      } else {
        lowConfidenceWarning.hidden = true;
      }

      // Grad-CAM images
      if (data.original_image_base64 && data.gradcam_image_base64) {
        gradcamOriginal.src = `data:image/png;base64,${data.original_image_base64}`;
        gradcamOverlay.src = `data:image/png;base64,${data.gradcam_image_base64}`;
        gradcamSection.hidden = false;
      } else {
        gradcamSection.hidden = true;
      }

      // Top-3 predictions
      if (data.top3 && data.top3.length > 0) {
        top3List.innerHTML = "";
        data.top3.forEach((item, index) => {
          const conf = item.confidence <= 1 ? item.confidence * 100 : item.confidence;
          const li = document.createElement("li");
          li.className = "top3-item";
          li.innerHTML = `
            <span class="top3-rank">${index + 1}</span>
            <span class="top3-label">${String(item.class).replaceAll("_", " ")}</span>
            <span class="top3-conf">${conf.toFixed(1)}%</span>
          `;
          top3List.appendChild(li);
        });
        top3Section.hidden = false;
      } else {
        top3Section.hidden = true;
      }

      loadingState.hidden = true;
      resultContent.hidden = false;
      resultIndicator.textContent = "Complete";
      resultIndicator.className = "result-indicator";
      showSuccess("Analysis complete. Grad-CAM heatmap and top-3 predictions shown below.");
    } catch (error) {
      loadingState.hidden = true;
      resultEmpty.hidden = false;
      resultIndicator.textContent = "Error";
      resultIndicator.className = "result-indicator";
      showError(
        `Could not get a prediction. ${error.message || "Check that the backend is running."}`
      );
    } finally {
      predictButton.disabled = !selectedFile;
      predictButtonLabel.textContent = "Analyze disease";
    }
  }

  // --- Event listeners ---
  // Auth tabs
  signinTab.addEventListener("click", () => switchAuthTab("signin"));
  signupTab.addEventListener("click", () => switchAuthTab("signup"));

  // Sign in
  signinForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = signinForm.email.value.trim().toLowerCase();
    const password = signinForm.password.value;
    const users = getUsers();
    const user = users.find((u) => u.email === email && u.password === password);
    if (user) {
      setCurrentUser({ name: user.name, email: user.email });
      signinForm.reset();
    } else {
      showAuthMessage("Invalid email or password.", "error");
    }
  });

  // Sign up
  signupForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = signupForm.name.value.trim();
    const email = signupForm.email.value.trim().toLowerCase();
    const password = signupForm.password.value;

    if (password.length < 8) {
      showAuthMessage("Password must be at least 8 characters.", "error");
      return;
    }

    const users = getUsers();
    if (users.some((u) => u.email === email)) {
      showAuthMessage("An account with this email already exists.", "error");
      return;
    }

    users.push({ name, email, password });
    saveUsers(users);
    setCurrentUser({ name, email });
    signupForm.reset();
    showAuthMessage("Account created successfully!", "success");
  });

  // Sign out
  signoutButton.addEventListener("click", () => {
    setCurrentUser(null);
  });

  // App tabs
  appTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  // Overview quick action
  overviewAnalyzeBtn.addEventListener("click", () => switchTab("crop-disease"));

  // File upload
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", (event) => setFile(event.target.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    setFile(file);
  });

  removeButton.addEventListener("click", clearSelectedFile);
  dismissError.addEventListener("click", () => { errorBanner.hidden = true; });

  // Form submit
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      showError("Please choose a leaf image before predicting.");
      return;
    }
    await predictDisease(selectedFile);
  });

  window.addEventListener("beforeunload", clearPreviewUrl);

  // --- Init ---
  const savedUser = getCurrentUser();
  if (savedUser) {
    setCurrentUser(savedUser);
  } else {
    showAuth();
  }
})();