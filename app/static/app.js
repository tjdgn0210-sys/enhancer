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
const cleanedResult = document.querySelector("#cleaned-result");
const cleanedResultImage = document.querySelector("#cleaned-result-image");
const resultImage = document.querySelector("#result-image");
const downloadResult = document.querySelector("#download-result");
const resultHeading = result.querySelector("h2");
const textOverlay = document.querySelector("#text-overlay");
const textCount = document.querySelector("#text-count");
const textList = document.querySelector("#text-list");
const selectExtraPoints = document.querySelector("#select-extra-points");
const extraPointControls = document.querySelector("#extra-point-controls");
const extraPointActions = document.querySelector("#extra-point-actions");
const undoPoint = document.querySelector("#undo-point");
const clearPointsButton = document.querySelector("#clear-points");
const pointStatus = document.querySelector("#point-status");
const maxFileSize = 20 * 1024 * 1024;
const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
let previewUrl = null;
let resultUrl = null;
let cleanedResultUrl = null;
let extraPoints = [];

function clearResult() {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  if (cleanedResultUrl) URL.revokeObjectURL(cleanedResultUrl);
  resultUrl = null;
  cleanedResultUrl = null;
  cleanedResultImage.removeAttribute("src");
  cleanedResult.hidden = true;
  resultImage.removeAttribute("src");
  downloadResult.removeAttribute("href");
  downloadResult.download = "enhanced-image.png";
  resultHeading.textContent = "Enhanced image";
  result.hidden = true;
}

function showCleanedResult(encodedImage) {
  const bytes = Uint8Array.from(atob(encodedImage), (character) => character.charCodeAt(0));
  cleanedResultUrl = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
  cleanedResultImage.src = cleanedResultUrl;
  cleanedResult.hidden = false;
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

function drawExtraPoints() {
  textOverlay.querySelectorAll(".extra-point-marker").forEach((marker) => marker.remove());
  if (previewImage.naturalWidth && previewImage.getBoundingClientRect().width) {
    for (const [index, point] of extraPoints.entries()) {
      const scale = previewImage.naturalWidth / previewImage.getBoundingClientRect().width;
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", point.x);
      circle.setAttribute("cy", point.y);
      circle.setAttribute("r", String(11 * scale));
      circle.classList.add("extra-point-marker");
      textOverlay.append(circle);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", point.x);
      label.setAttribute("y", point.y);
      label.setAttribute("font-size", String(11 * scale));
      label.textContent = String(index + 1);
      label.classList.add("extra-point-number");
      textOverlay.append(label);
    }
  }
  undoPoint.disabled = extraPoints.length === 0;
  clearPointsButton.disabled = extraPoints.length === 0;
  pointStatus.textContent = extraPoints.length ? "Selected UI points are ready for segmentation." : "";
}

function clearExtraPoints() {
  extraPoints = [];
  drawExtraPoints();
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
  clearExtraPoints();
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
    drawExtraPoints();
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
  extraPointControls.hidden = !enabled;
  extraPointActions.hidden = !enabled || !selectExtraPoints.checked;
}

mode.addEventListener("change", updateCleanupAvailability);
updateCleanupAvailability();

previewImage.addEventListener("click", (event) => {
  if (!selectExtraPoints.checked || mode.value !== "clean-enhance" || extraPoints.length >= 20) return;
  const rect = previewImage.getBoundingClientRect();
  const displayX = event.clientX - rect.left;
  const displayY = event.clientY - rect.top;
  if (displayX < 0 || displayY < 0 || displayX >= rect.width || displayY >= rect.height) return;
  const point = {
    x: Math.min(previewImage.naturalWidth - 1, Math.floor(displayX * previewImage.naturalWidth / rect.width)),
    y: Math.min(previewImage.naturalHeight - 1, Math.floor(displayY * previewImage.naturalHeight / rect.height)),
  };
  if (extraPoints.some((existing) => existing.x === point.x && existing.y === point.y)) return;
  extraPoints.push(point);
  drawExtraPoints();
});

selectExtraPoints.addEventListener("change", updateCleanupAvailability);
undoPoint.addEventListener("click", () => {
  extraPoints.pop();
  drawExtraPoints();
});
clearPointsButton.addEventListener("click", clearExtraPoints);
window.addEventListener("resize", drawExtraPoints);

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
  if (mode.value === "clean-enhance") data.append("extra_points", JSON.stringify(extraPoints));

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
      drawExtraPoints();
      showCleanedResult(data.cleaned_image);
      const enhancedBytes = Uint8Array.from(atob(data.enhanced_image), (character) => character.charCodeAt(0));
      showImageResult(
        new Blob([enhancedBytes], { type: "image/png" }),
        "cleaned-enhanced-image.png",
        "Final enhanced image",
      );
      status.textContent = `${data.message} (${data.enhanced_width} × ${data.enhanced_height})`;
      if (data.extra_points?.length) status.textContent += " Selected UI points are ready for segmentation.";
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
  clearExtraPoints();
  clearPreview();
  clearTextRegions();
  clearResult();
  uploadMessage.textContent = "";
  status.textContent = "";
  updateCleanupAvailability();
});
