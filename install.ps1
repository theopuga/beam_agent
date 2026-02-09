Write-Host "Beam Node Agent Quickstart"
Write-Host "This will download the agent, write a config, and start pairing."
Write-Host "Keep this terminal open to see the pairing code."
Write-Host ""

$confirm = Read-Host "Continue? [Y/n]"
if ([string]::IsNullOrWhiteSpace($confirm)) { $confirm = "Y" }
if ($confirm -notmatch "^[Yy]") {
  Write-Host "Aborted."
  exit 1
}

$controlPlaneUrl = $env:BEAM_CONTROL_PLANE_URL
if ([string]::IsNullOrWhiteSpace($controlPlaneUrl)) { $controlPlaneUrl = "https://beam-production-f317.up.railway.app" }
$controlPlaneUrl = $controlPlaneUrl.TrimEnd("/")
Write-Host "Control plane URL: $controlPlaneUrl"

$releaseBaseDefault = "https://github.com/theopuga/beam_agent/releases/latest/download"
$releaseBase = Read-Host "Release base URL [$releaseBaseDefault]"
if ([string]::IsNullOrWhiteSpace($releaseBase)) { $releaseBase = $releaseBaseDefault }
$releaseBase = $releaseBase.TrimEnd("/")

$assetName = "beam-node-agent-windows.exe"
$binaryPath = ".\$assetName"
$downloadUrl = "$releaseBase/$assetName"

if (Test-Path $binaryPath) {
  $redownload = Read-Host "$binaryPath exists. Redownload? [y/N]"
  if ([string]::IsNullOrWhiteSpace($redownload)) { $redownload = "N" }
  if ($redownload -match "^[Yy]") {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $binaryPath
    Write-Host "Downloaded $binaryPath"
  } else {
    Write-Host "Using existing $binaryPath"
  }
} else {
  Invoke-WebRequest -Uri $downloadUrl -OutFile $binaryPath
  Write-Host "Downloaded $binaryPath"
}

$configPath = Read-Host "Config path [config.yaml] (press Enter for default)"
if ([string]::IsNullOrWhiteSpace($configPath)) { $configPath = "config.yaml" }
if ($configPath -match '^(?i)(y|yes|n|no)$') {
  Write-Host "Using default config.yaml"
  $configPath = "config.yaml"
}

if (Test-Path $configPath) {
  $overwrite = Read-Host "$configPath exists. Overwrite? [y/N]"
  if ([string]::IsNullOrWhiteSpace($overwrite)) { $overwrite = "N" }
  if ($overwrite -match "^[Yy]") {
    @"
control_plane:
  url: "$controlPlaneUrl"

petals:
  port: 31337
  gpu_vram_limit: 0.9

agent:
  heartbeat_interval_sec: 15
  state_file: "node_state.json"
  transports:
    - "fast"
  pairing_host: "127.0.0.1"
  pairing_ports:
    - 51337
    - 51338
    - 51339
    - 51340
  capabilities:
    supports_heavy_middle_layers: true
    max_concurrent_jobs: 1
"@ | Set-Content -Path $configPath -Encoding UTF8
    Write-Host "Wrote $configPath"
  } else {
    Write-Host "Using existing $configPath"
  }
} else {
  @"
control_plane:
  url: "$controlPlaneUrl"

petals:
  port: 31337
  gpu_vram_limit: 0.9

agent:
  heartbeat_interval_sec: 15
  state_file: "node_state.json"
  transports:
    - "fast"
  pairing_host: "127.0.0.1"
  pairing_ports:
    - 51337
    - 51338
    - 51339
    - 51340
  capabilities:
    supports_heavy_middle_layers: true
    max_concurrent_jobs: 1
"@ | Set-Content -Path $configPath -Encoding UTF8
  Write-Host "Wrote $configPath"
}

Write-Host ""
Write-Host "Next: the agent will start and print a 6-digit pair code."
Write-Host "Open the Rent Panel and enter the code to link this machine."
Write-Host ""

$startNow = Read-Host "Start the agent now? [Y/n]"
if ([string]::IsNullOrWhiteSpace($startNow)) { $startNow = "Y" }
if ($startNow -notmatch "^[Yy]") {
  Write-Host "You can start it later with:"
  Write-Host "$binaryPath --config $configPath"
  exit 0
}

Write-Host "Running: $binaryPath --config $configPath"
$env:BEAM_CONTROL_PLANE_URL = $controlPlaneUrl
$env:CONTROL_PLANE_URL = $controlPlaneUrl
& $binaryPath --config $configPath
