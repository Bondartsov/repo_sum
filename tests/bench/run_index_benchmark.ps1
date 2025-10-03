param(
    [Parameter(Mandatory=$true)][string]$IndexCmd,
    [Parameter(Mandatory=$true)][string]$MetricsDir,
    [string[]]$Env = @(),
    [string]$Profile = "baseline",
    [string]$Tag = $null
)

# Ensure metrics dir
New-Item -ItemType Directory -Force -Path $MetricsDir | Out-Null

# Prepare environment overrides
$envOverrides = @()
foreach ($e in $Env) {
    if ($e -notmatch "=") { throw "Bad --Env item: $e" }
    $envOverrides += "--env `"$e`""
}

# Build python command
$cmd = "python .\run_ab.py --index-cmd `"$IndexCmd`" --metrics-dir `"$MetricsDir`" --profile $Profile " + ($envOverrides -join " ")
if ($Tag) { $cmd += " --tag `"$Tag`"" }

Write-Host "Running: $cmd"
cmd.exe /c $cmd
