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
const resultHeading = result.querySelector("h2");
const textOverlay = document.querySelector("#text-overlay");
const textCount = document.querySelector("#text-count");
const textList = document.querySelector("#text-list");
const maxFileSize = 20 * 1024 * 1024;
const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
let previewUrl = null;
let resultUrl = null;

function clearResult() {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = null;
  resultImage.removeAttribute("src");
  downloadResult.removeAttribute("href");
  downloadResult.download = "enhanced-image.png";
  resultHeading.textContent = "Enhanced image";
  result.hidden = true;
}

function showImageResult(imageBlob, filename, heading) {
  resultUrl = URL.createObjectURL(imageBlob);
  resultImage.src = resultUrl;
  downloadResult.href = resultUrl;
  downloadResult.download = filename;
  resultHeading.textContent = heading;
  result.hidden = false;
}

function clearTextRegions() {
  textOverlay.replaceChildren();
  textOverlay.removeAttribute("viewBox");
  textCount.textContent = "";
  textCount.hidden = true;
  textList.replaceChildren();
  textList.hidden = true;
}

function showTextRegions(regions, width, height, maskPreview) {
  clearTextRegions();
  textOverlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (maskPreview) {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const mask = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    mask.id = "removal-mask";
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    mask.setAttribute("maskContentUnits", "userSpaceOnUse");
    mask.setAttribute("mask-type", "luminance");
    mask.setAttribute("x", "0");
    mask.setAttribute("y", "0");
    mask.setAttribute("width", width);
    mask.setAttribute("height", height);
    const maskImage = document.createElementNS("http://www.w3.org/2000/svg", "image");
    maskImage.setAttribute("href", `data:image/png;base64,${maskPreview}`);
    maskImage.setAttribute("width", width);
    maskImage.setAttribute("height", height);
    maskImage.setAttribute("preserveAspectRatio", "none");
    mask.append(maskImage);
    defs.append(mask);
    textOverlay.append(defs);

    const highlightedMask = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    highlightedMask.setAttribute("width", width);
    highlightedMask.setAttribute("height", height);
    highlightedMask.setAttribute("fill", "#ef4444");
    highlightedMask.setAttribute("fill-opacity", "0.38");
    highlightedMask.setAttribute("mask", "url(#removal-mask)");
    textOverlay.append(highlightedMask);
  }
  for (const region of regions) {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", region.polygon.map((point) => point.join(",")).join(" "));
    polygon.classList.add("text-region");
    textOverlay.append(polygon);

    const item = document.createElement("li");
    item.textContent = `${region.text} (${(region.confidence * 100).toFixed(0)}%)`;
    textList.append(item);
  }
  textCount.textContent = `Detected text: ${regions.length}`;
  textCount.hidden = false;
  textList.hidden = regions.length === 0;
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
  clearTextRegions();
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
  clearTextRegions();
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
    if (mode.value === "clean-enhance") {
      const data = await response.json();
      showTextRegions(data.text_regions || [], data.width, data.height, data.mask_preview);
      const imageBytes = Uint8Array.from(atob(data.cleaned_image), (character) => character.charCodeAt(0));
      showImageResult(new Blob([imageBytes], { type: "image/png" }), "cleaned-image.png", "Cleaned image");
      status.textContent = `${data.message} (${data.width} × ${data.height})`;
    } else if (response.headers.get("content-type")?.startsWith("image/")) {
      const output = await response.blob();
      showImageResult(output, "enhanced-image.png", "Enhanced image");
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
  clearTextRegions();
  clearResult();
  uploadMessage.textContent = "";
  status.textContent = "";
  updateCleanupAvailability();
});
