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

# Keep-awake. The scan takes 2 to 4 minutes and the laptop was going back to sleep
# partway through, killing the task with exit 0xC000013A and leaving a zero-byte log
# (08-04, and the same mode on 07-26 and 08-03). SetThreadExecutionState tells Windows
# a job is in progress so the sleep timer does not fire. ES_CONTINUOUS keeps the request
# alive until it is cleared, ES_SYSTEM_REQUIRED blocks sleep, ES_AWAYMODE_REQUIRED covers
# the away mode used on some sleep settings. The display is deliberately left alone, so
# the screen can still switch off. Cleared in the finally block below.
$sig = @"
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
try { Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -ErrorAction Stop } catch {}
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"

try {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null
} catch {}

try {
    & $python dividend_scanner_v10.py *> $log
    $code = $LASTEXITCODE
} finally {
    # Hand sleep control back to Windows whether the scan succeeded or not.
    try { [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null } catch {}
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$csv   = "candidates_$(Get-Date -Format yyyyMMdd).csv"
if ($code -eq 0 -and (Test-Path $csv)) {
    "OK   $stamp  scan finished, $csv written" | Set-Content "LAST_RUN.txt" -Encoding utf8
} else {
    "FAIL $stamp  exit code $code, no candidates file" | Set-Content "LAST_RUN.txt" -Encoding utf8
}
