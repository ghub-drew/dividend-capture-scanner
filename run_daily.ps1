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

# Run once a day, whichever trigger gets here first. The task now has two: the 8:00
# daily one, and a logon trigger five minutes after sign-in that covers the mornings
# the laptop was off at 8:00. Whichever fires first does the scan and the other exits
# here. Without this guard every sign-in would start another full 576-ticker scan,
# and back-to-back scans are what triggered the Yahoo "401 Invalid Crumb" block on 07-30.
if (Test-Path "candidates_$(Get-Date -Format yyyyMMdd).csv") {
    "SKIP $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  already scanned today" | Add-Content "logs\trigger.log" -Encoding utf8
    exit 0
}

# Full path on purpose. Plain "python" resolves to the Windows Store app alias in
# AppData\Local\Microsoft\WindowsApps, which works in an interactive shell but is
# killed instantly (exit 0xC000013A) when a scheduled task launches it. That was
# the cause of the zero-byte logs on 06-14, 06-17, 06-21, 06-28, 07-26, 07-29 and 07-30.
$python = "C:\Users\andre\AppData\Local\Python\pythoncore-3.14-64\python.exe"

# Keep-awake, so a long scan cannot be cut short by the sleep timer.
#
# Note on the zero-byte logs this was originally aimed at: it was NOT the cause.
# Every failed run wrote its log 2 to 3 seconds after the trigger fired, far too fast
# for a scan that takes 2 to 4 minutes to have been interrupted partway. Those runs
# were killed on startup, not mid-scan: the task runs as an Interactive logon, so it
# needs a resumed user session, and it was being fired before one existed.
#
# Turning "wake to run" off on 08-28 did not close it. The real pattern is clearer:
# the runs that fired ON TIME at 08:01 with the machine already awake finished (08-29,
# 08-30), and every "start when available" CATCH-UP fired during sign-in was killed
# (08-31 09:47, 09-01 09:40, 09-02 09:28). The catch-up is the broken path, because it
# fires while the session is still coming up.
#
# Fixed 09-02 without admin rights (the proper fix, LogonType S4U, needs the batch
# logon right and is refused even elevated). "Start when available" is now OFF, so the
# 8:00 trigger only runs when the machine is already awake, and a second trigger runs
# five minutes AFTER sign-in, once the session is settled, to cover the mornings the
# laptop was off at 8:00. The same-day guard at the top keeps the two from both scanning.
#
# The keep-awake below stays because it guards a real and separate risk: the machine
# sleeping during the scan itself. SetThreadExecutionState tells Windows a job is in
# progress so the sleep timer does not fire. ES_CONTINUOUS keeps the request
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
