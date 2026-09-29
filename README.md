# Enhancer

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## Local enhancement engine

For Enhance Only, place `realesrgan-ncnn-vulkan.exe` in `tools/realesrgan/` and the `realesrgan-x4plus.param` / `realesrgan-x4plus.bin` model files in `tools/realesrgan/models/`. The executable and model files are not included.

Enhancement levels map to CLI output scales: level 1 = 2x, level 2 = 3x (recommended), level 3 = 4x, and level 4 = 4x (highest available scale). The CLI exposes scale, not a distinct strength setting; levels 3 and 4 therefore currently use the same setting.
