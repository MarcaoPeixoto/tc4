# Um comando: sobe API (treina se precisar + abre navegador)
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    python -m venv .venv
    & $py -m pip install -r requirements.txt -q
}
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Iniciando API em http://localhost:8765 (Ctrl+C para parar)" -ForegroundColor Cyan
& $py api.py
