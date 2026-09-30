$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$toolsDir = Join-Path $repoRoot "tools"
$realesrganDir = Join-Path $toolsDir "realesrgan"
$realesrganModelsDir = Join-Path $realesrganDir "models"
$lamaDir = Join-Path $toolsDir "lama"
$segmentationDir = Join-Path $toolsDir "segmentation"

$realesrganArchiveUrl = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
$lamaModelUrl = "https://huggingface.co/opencv/inpainting_lama/resolve/c47be270660574005b6c951b5904f51a1af28f6d/inpainting_lama_2025jan.onnx?download=true"

foreach ($directory in @($realesrganDir, $realesrganModelsDir, $lamaDir, $segmentationDir)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

function Test-UsableFile([string] $path) {
    return (Test-Path -LiteralPath $path -PathType Leaf) -and ((Get-Item -LiteralPath $path).Length -gt 0)
}

function Copy-ArchiveEntry([System.IO.Compression.ZipArchiveEntry] $entry, [string] $destination) {
    $temporaryPath = "$destination.download"
    try {
        $inputStream = $entry.Open()
        try {
            $outputStream = [System.IO.File]::Create($temporaryPath)
            try { $inputStream.CopyTo($outputStream) } finally { $outputStream.Dispose() }
        } finally { $inputStream.Dispose() }
        if ((Get-Item -LiteralPath $temporaryPath).Length -eq 0) {
            throw "Archive entry '$($entry.FullName)' was empty."
        }
        Move-Item -LiteralPath $temporaryPath -Destination $destination -Force
    } finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
}

$archiveFiles = @(
    @{ Name = "realesrgan-ncnn-vulkan.exe"; RelativePath = "tools/realesrgan/realesrgan-ncnn-vulkan.exe"; Destination = (Join-Path $realesrganDir "realesrgan-ncnn-vulkan.exe") },
    @{ Name = "vcomp140.dll"; RelativePath = "tools/realesrgan/vcomp140.dll"; Destination = (Join-Path $realesrganDir "vcomp140.dll") },
    @{ Name = "vcomp140d.dll"; RelativePath = "tools/realesrgan/vcomp140d.dll"; Destination = (Join-Path $realesrganDir "vcomp140d.dll") },
    @{ Name = "realesrgan-x4plus.param"; RelativePath = "tools/realesrgan/models/realesrgan-x4plus.param"; Destination = (Join-Path $realesrganModelsDir "realesrgan-x4plus.param") },
    @{ Name = "realesrgan-x4plus.bin"; RelativePath = "tools/realesrgan/models/realesrgan-x4plus.bin"; Destination = (Join-Path $realesrganModelsDir "realesrgan-x4plus.bin") }
)
$missingArchiveFiles = @($archiveFiles | Where-Object { -not (Test-UsableFile $_.Destination) })

if ($missingArchiveFiles.Count -gt 0) {
    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("enhancer-setup-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    $archivePath = Join-Path $temporaryRoot "realesrgan-windows.zip"
    try {
        Write-Host "Downloading the official Real-ESRGAN Windows release..."
        Invoke-WebRequest -Uri $realesrganArchiveUrl -OutFile $archivePath -UseBasicParsing
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
        try {
            foreach ($file in $missingArchiveFiles) {
                $entry = $archive.Entries | Where-Object { [System.IO.Path]::GetFileName($_.FullName) -ceq $file.Name } | Select-Object -First 1
                if ($null -eq $entry) {
                    throw "Required file '$($file.Name)' was not found in the Real-ESRGAN release archive."
                }
                Write-Host "Restoring $($file.RelativePath)..."
                Copy-ArchiveEntry $entry $file.Destination
            }
        } finally {
            $archive.Dispose()
        }
    } catch {
        throw "Could not restore Real-ESRGAN files: $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Real-ESRGAN runtime and model files are already present; skipping download."
}

$lamaModelPath = Join-Path $lamaDir "inpainting_lama_2025jan.onnx"
if (Test-UsableFile $lamaModelPath) {
    Write-Host "LaMa ONNX model is already present; skipping download."
} else {
    $temporaryModelPath = Join-Path ([System.IO.Path]::GetTempPath()) ("inpainting_lama_" + [guid]::NewGuid().ToString("N") + ".onnx")
    try {
        Write-Host "Downloading the OpenCV LaMa ONNX model..."
        Invoke-WebRequest -Uri $lamaModelUrl -OutFile $temporaryModelPath -UseBasicParsing
        if (-not (Test-UsableFile $temporaryModelPath) -or (Get-Item -LiteralPath $temporaryModelPath).Length -lt 1000000) {
            throw "The downloaded model is missing or unexpectedly small."
        }
        Move-Item -LiteralPath $temporaryModelPath -Destination $lamaModelPath -Force
        Write-Host "Restored tools/lama/inpainting_lama_2025jan.onnx."
    } catch {
        throw "Could not restore the LaMa ONNX model: $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $temporaryModelPath -Force -ErrorAction SilentlyContinue
    }
}

$edgeSamRevision = "5d010ec34cef497a7205e375222647e2d2f57091"
$segmentationModels = @(
    @{
        Name = "edge_sam_3x_encoder.onnx"
        Sha256 = "719a498cf5b3fe9be9f01ee513e13d3915f9028aa4f23dfd30eaaa0a17143159"
        Url = "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/$edgeSamRevision/weights/edge_sam_3x_encoder.onnx?download=true"
    },
    @{
        Name = "edge_sam_3x_decoder.onnx"
        Sha256 = "83a2174d54571596913dcb7455d021e713623c3dca30a31c8c41ab98c9fb0863"
        Url = "https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/$edgeSamRevision/weights/edge_sam_3x_decoder.onnx?download=true"
    }
)
$missingSegmentationModels = @($segmentationModels | Where-Object {
    -not (Test-UsableFile (Join-Path $segmentationDir $_.Name))
})

if ($missingSegmentationModels.Count -gt 0) {
    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("edgesam-setup-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    try {
        foreach ($model in $missingSegmentationModels) {
            $temporaryModelPath = Join-Path $temporaryRoot $model.Name
            $destination = Join-Path $segmentationDir $model.Name
            Write-Host "Downloading EdgeSAM model $($model.Name)..."
            Invoke-WebRequest -Uri $model.Url -OutFile $temporaryModelPath -UseBasicParsing
            if (-not (Test-UsableFile $temporaryModelPath)) {
                throw "Downloaded EdgeSAM model '$($model.Name)' is missing or empty."
            }
            $actualHash = (Get-FileHash -LiteralPath $temporaryModelPath -Algorithm SHA256).Hash
            if ($actualHash -ne $model.Sha256) {
                throw "Downloaded EdgeSAM model '$($model.Name)' failed its SHA256 check."
            }
            Move-Item -LiteralPath $temporaryModelPath -Destination $destination -Force
        }
    } catch {
        throw "Could not restore EdgeSAM ONNX models: $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "EdgeSAM ONNX models are already present; skipping download."
}

Write-Host "Development runtime setup complete."
Write-Host "Next, create the Python environment and install dependencies:"
Write-Host "  python -m venv .venv"
Write-Host "  .\.venv\Scripts\python.exe -m pip install --upgrade pip"
Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
