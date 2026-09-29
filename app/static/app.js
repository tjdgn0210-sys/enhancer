const form = document.querySelector("#process-form");
const mode = document.querySelector("#mode");
const cleanupGroup = document.querySelector("#cleanup-group");
const cleanupLevel = document.querySelector("#cleanup-level");
const status = document.querySelector("#status");
const imageInput = document.querySelector("#image");
const uploadMessage = document.querySelector("#upload-message");
const preview = document.querySelector("#preview");
const previewImage = document.querySelector("#preview-image");
const imageName = document.querySelector("#image-name");
const imageSize = document.querySelector("#image-size");
const imageDimensions = document.querySelector("#image-dimensions");
const result = document.querySelector("#result");
const resultImage = document.querySelector("#result-image");
const downloadResult = document.querySelector("#download-result");
const maxFileSize = 20 * 1024 * 1024;
const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
let previewUrl = null;
let resultUrl = null;

function clearResult() {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = null;
  resultImage.removeAttribute("src");
  downloadResult.removeAttribute("href");
  result.hidden = true;
}

function clearPreview() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImage.removeAttribute("src");
  preview.hidden = true;
  imageName.textContent = "";
  imageSize.textContent = "";
  imageDimensions.textContent = "";
}

imageInput.addEventListener("change", () => {
  clearPreview();
  uploadMessage.textContent = "";
  status.textContent = "";
  const image = imageInput.files[0];
  if (!image) return;

  const extension = image.name.split(".").pop().toLowerCase();
  const validExtension = ["jpg", "jpeg", "png", "webp"].includes(extension);
  if (!allowedTypes.has(image.type) || !validExtension) {
    imageInput.value = "";
    uploadMessage.textContent = "Unsupported file type. Choose a JPG, PNG, or WebP image.";
    return;
  }
  if (image.size > maxFileSize) {
    imageInput.value = "";
    uploadMessage.textContent = "Image is too large. Maximum file size is 20 MB.";
    return;
  }

  previewUrl = URL.createObjectURL(image);
  previewImage.onload = () => {
    imageName.textContent = image.name;
    imageSize.textContent = `${(image.size / (1024 * 1024)).toFixed(2)} MB`;
    imageDimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight}`;
    preview.hidden = false;
  };
  previewImage.onerror = () => {
    clearPreview();
    imageInput.value = "";
    uploadMessage.textContent = "The selected file could not be read as an image.";
  };
  previewImage.src = previewUrl;
});

function updateCleanupAvailability() {
  const enabled = mode.value === "clean-enhance";
  cleanupGroup.hidden = !enabled;
  cleanupLevel.disabled = !enabled;
}

mode.addEventListener("change", updateCleanupAvailability);
updateCleanupAvailability();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const image = document.querySelector("#image").files[0];
  if (!image) {
    status.textContent = "Choose an image before processing.";
    return;
  }
  const extension = image.name.split(".").pop().toLowerCase();
  if (!allowedTypes.has(image.type) || !["jpg", "jpeg", "png", "webp"].includes(extension) || image.size > maxFileSize) {
    status.textContent = "Choose a JPG, PNG, or WebP image up to 20 MB.";
    return;
  }

  const processButton = form.querySelector('button[type="submit"]');
  clearResult();
  const data = new FormData();
  data.append("image", image);
  data.append("mode", mode.value.replace("-", "_"));
  data.append("enhancement_level", document.querySelector("#enhancement-level").value);
  if (mode.value === "clean-enhance") data.append("cleanup_level", cleanupLevel.value);

  processButton.disabled = true;
  status.textContent = "Processing...";
  uploadMessage.textContent = "";
  try {
    const response = await fetch("/api/process", { method: "POST", body: data });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Image processing failed.");
    }
    if (response.headers.get("content-type")?.startsWith("image/")) {
      const output = await response.blob();
      resultUrl = URL.createObjectURL(output);
      resultImage.src = resultUrl;
      downloadResult.href = resultUrl;
      result.hidden = false;
      status.textContent = `Enhanced image ready (${response.headers.get("X-Image-Width")} × ${response.headers.get("X-Image-Height")}).`;
    } else {
      const message = await response.json();
      status.textContent = message.message || "Image processing is not available yet.";
    }
  } catch (error) {
    status.textContent = error.message || "Could not connect to the server. Please try again.";
  } finally {
    processButton.disabled = false;
  }
});

document.querySelector("#reset").addEventListener("click", () => {
  form.reset();
  clearPreview();
  clearResult();
  uploadMessage.textContent = "";
  status.textContent = "";
  updateCleanupAvailability();
});
