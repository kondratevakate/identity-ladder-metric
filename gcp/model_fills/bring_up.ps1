# Bring a GPU VM back for the model-fill runs after a spot preemption (identity ladder, PROTOCOL_model_fills.md).
# Order: the stopped VM in its own zone (no data transfer), then a new VM from the snapshot, zones cheapest first by
# the on-demand catalog rate of 16 Nov 2026; in every zone spot is tried before on-demand. L4 first, T4 as fallback.
# Prints "VM=<name> ZONE=<zone> MODEL=<SPOT|STANDARD> GPU=<l4|t4>" on success and exits 1 if no zone has capacity.
$ErrorActionPreference = "Continue"
$g = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
$old = "ladder-modelfills"; $oldZone = "europe-west4-a"; $snap = "ladder-modelfills-snap"

function Status($name, $zone) { (& $g compute instances describe $name --zone=$zone --format="value(status)" 2>$null) }

# A. the stopped VM, spot then on-demand
foreach ($model in "SPOT", "STANDARD") {
  if ($model -eq "SPOT") {
    & $g compute instances set-scheduling $old --zone=$oldZone --preemptible --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE 2>&1 | Out-Null
  } else {
    & $g compute instances set-scheduling $old --zone=$oldZone --no-preemptible --provisioning-model=STANDARD --clear-instance-termination-action --maintenance-policy=TERMINATE 2>&1 | Out-Null
  }
  & $g compute instances start $old --zone=$oldZone 2>&1 | Out-Null
  if ((Status $old $oldZone) -eq "RUNNING") { Write-Output "VM=$old ZONE=$oldZone MODEL=$model GPU=l4"; exit 0 }
  Write-Output "no capacity: $old $oldZone $model"
}

# B. a new VM from the snapshot
$l4 = "europe-west4-b", "europe-west4-c", "europe-west1-b", "europe-west1-c", "europe-west2-a", "europe-west2-b", "europe-west9-a", "europe-west9-b", "europe-west9-c", "europe-west3-a", "europe-west3-b", "europe-southwest1-a", "europe-southwest1-b", "europe-west6-a", "europe-west6-b"
$t4 = "europe-west4-a", "europe-west4-b", "europe-west4-c", "europe-west1-b", "europe-west1-c", "europe-west1-d", "europe-west2-a", "europe-west2-b", "europe-west3-b"
$plans = @()
foreach ($z in $l4) { $plans += , @($z, "g2-standard-8", "nvidia-l4", "l4") }
foreach ($z in $t4) { $plans += , @($z, "n1-standard-8", "nvidia-tesla-t4", "t4") }
foreach ($p in $plans) {
  $z, $mt, $acc, $tag = $p
  foreach ($model in "SPOT", "STANDARD") {
    $extra = @()
    if ($model -eq "SPOT") { $extra = @("--provisioning-model=SPOT", "--instance-termination-action=STOP") }
    & $g compute instances create ladder-modelfills2 --zone=$z --machine-type=$mt --accelerator="type=$acc,count=1" --maintenance-policy=TERMINATE @extra --create-disk="boot=yes,source-snapshot=$snap,size=150,type=pd-balanced,auto-delete=yes" --labels="owner=model-fills,project=identity-ladder" 2>&1 | Out-Null
    if ((Status "ladder-modelfills2" $z) -eq "RUNNING") { Write-Output "VM=ladder-modelfills2 ZONE=$z MODEL=$model GPU=$tag"; exit 0 }
    Write-Output "no capacity: $z $mt $model"
  }
}
Write-Output "NO_CAPACITY"
exit 1
