# Runs the dividend scanner and saves the output to a dated log file.
# Used by the "DividendScannerDaily" scheduled task for the morning auto-run.
#
# Kill switch (pause / resume the morning run):
#   schtasks /change /tn "DividendScannerDaily" /disable
#   schtasks /change /tn "DividendScannerDaily" /enable
#
# LAST_RUN.txt is the tripwire: it records the finish time and the result of the
# most recent run, so a task that dies mid-scan no longer looks like success.
Set-Location $PSScriptRoot
if (-not (Test-Path "logs")) { New-Item -ItemType Directory "logs" | Out-Null }
$log = "logs\scan_$(Get-Date -Format yyyyMMdd).log"

# Full path on purpose. Plain "python" resolves to the Windows Store app alias in
# AppData\Local\Microsoft\WindowsApps, which works in an interactive shell but is
# killed instantly (exit 0xC000013A) when a scheduled task launches it. That was
# the cause of the zero-byte logs on 06-14, 06-17, 06-21, 06-28, 07-26, 07-29 and 07-30.
$python = "C:\Users\andre\AppData\Local\Python\pythoncore-3.14-64\python.exe"

& $python dividend_scanner_v9.py *> $log
$code = $LASTEXITCODE

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$csv   = "candidates_$(Get-Date -Format yyyyMMdd).csv"
if ($code -eq 0 -and (Test-Path $csv)) {
    "OK   $stamp  scan finished, $csv written" | Set-Content "LAST_RUN.txt" -Encoding utf8
} else {
    "FAIL $stamp  exit code $code, no candidates file" | Set-Content "LAST_RUN.txt" -Encoding utf8
}
