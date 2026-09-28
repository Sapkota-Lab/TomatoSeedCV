<#
Builds a standalone Windows executable for TomatoSeedCV using PyInstaller.
Run from the project root:  ./build_exe.ps1
Output: dist/TomatoSeedCV/TomatoSeedCV.exe (and its supporting files/folder).
#>

pip install pyinstaller | Out-Null

$inferenceSdkArgs = @()
python -c "import inference_sdk" 2>$null
if ($LASTEXITCODE -eq 0) {
    $inferenceSdkArgs = @("--collect-all", "inference_sdk")
} else {
    Write-Host "inference_sdk not installed in this environment (skipping bisected-seed support in this build)."
}

python -m PyInstaller `
  --name TomatoSeedCV `
  --noconfirm `
  --collect-all shiny `
  --collect-all htmltools `
  --collect-all cv2 `
  @inferenceSdkArgs `
  --exclude-module torch `
  --exclude-module torchvision `
  --exclude-module scipy `
  --exclude-module matplotlib `
  --exclude-module pandas `
  --exclude-module sympy `
  --exclude-module lxml `
  --exclude-module psycopg_binary `
  --exclude-module IPython `
  --exclude-module jupyter `
  --hidden-import uvicorn.logging `
  --hidden-import uvicorn.loops `
  --hidden-import uvicorn.loops.auto `
  --hidden-import uvicorn.protocols `
  --hidden-import uvicorn.protocols.http `
  --hidden-import uvicorn.protocols.http.auto `
  --hidden-import uvicorn.protocols.websockets `
  --hidden-import uvicorn.protocols.websockets.auto `
  --hidden-import uvicorn.lifespan `
  --hidden-import uvicorn.lifespan.on `
  --add-data "src;src" `
  desktop_app.py


Write-Host ""
Write-Host "Build complete. Share the dist/TomatoSeedCV folder (zipped) with others."
Write-Host "They must place a .env file with ROBOFLOW_API_KEY next to TomatoSeedCV.exe."
