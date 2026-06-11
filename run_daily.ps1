# Runs the dividend scanner and saves the output to a dated log file.
# Used by the "DividendScannerDaily" scheduled task for the morning auto-run.
#
# Kill switch (pause / resume the morning run):
#   schtasks /change /tn "DividendScannerDaily" /disable
#   schtasks /change /tn "DividendScannerDaily" /enable
Set-Location $PSScriptRoot
if (-not (Test-Path "logs")) { New-Item -ItemType Directory "logs" | Out-Null }
$log = "logs\scan_$(Get-Date -Format yyyyMMdd).log"
python dividend_scanner_v6.py *> $log
