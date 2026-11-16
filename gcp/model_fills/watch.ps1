# Watch the model-fill VM: restart it after a spot preemption (spot first, then on-demand), re-run the idempotent
# resume script, and exit when the marker appears in ~/mf/queue.log or ~/mf/queue_brain.log, or when both runs are done.
# Usage: watch.ps1 -Marker "<fixed string>" [-MaxMinutes 600]
param([string]$Marker = "ALL_DONE", [int]$MaxMinutes = 600, [string]$Vm = "ladder-modelfills", [string]$Zone = "europe-west4-a")
$g = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$t0 = Get-Date
while (((Get-Date) - $t0).TotalMinutes -lt $MaxMinutes) {
  $st = & $g compute instances describe $Vm --zone=$Zone --format="value(status)" 2>$null
  if ($st -ne "RUNNING") {
    Write-Output ("{0:HH:mm} VM status {1}: restarting" -f (Get-Date), $st)
    & $g compute instances start $Vm --zone=$Zone 2>&1 | Out-Null
    $st = & $g compute instances describe $Vm --zone=$Zone --format="value(status)" 2>$null
    if ($st -ne "RUNNING") {      # no spot capacity: on-demand in the same zone, so the disk and its results stay
      & $g compute instances set-scheduling $Vm --zone=$Zone --no-preemptible --provisioning-model=STANDARD --clear-instance-termination-action --maintenance-policy=TERMINATE 2>&1 | Out-Null
      & $g compute instances start $Vm --zone=$Zone 2>&1 | Out-Null
      $st = & $g compute instances describe $Vm --zone=$Zone --format="value(status)" 2>$null
    }
    if ($st -ne "RUNNING") { Start-Sleep -Seconds 240; continue }
    Start-Sleep -Seconds 40
  }
  $out = & $g compute ssh $Vm --zone=$Zone --command="bash mf/resume.sh; cat mf/queue.log mf/queue_brain.log" 2>$null
  $txt = ($out | Out-String)
  if ($txt -match "ALL_DONE" -or $txt.Contains($Marker)) {
    Write-Output ("{0:HH:mm} marker reached: {1}" -f (Get-Date), $Marker)
    $out | Select-Object -Last 12
    exit 0
  }
  Start-Sleep -Seconds 240
}
Write-Output "TIMEOUT"
exit 2
