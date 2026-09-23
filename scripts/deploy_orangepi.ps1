[CmdletBinding()]
param(
    [string]$OrangePiHost = "orangepi5pro.local",
    [string]$UserName = "orangepi",
    [string]$RemoteDirectory = "/home/orangepi/dangerous_driving_vision"
)

$ErrorActionPreference = "Stop"

if ($OrangePiHost -notmatch '^[A-Za-z0-9._:-]+$') {
    throw "OrangePiHost contains unsupported characters."
}
if ($UserName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "UserName contains unsupported characters."
}
if ($RemoteDirectory -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "RemoteDirectory must be an absolute Linux path."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) (
    "dangerous-driving-{0}.tar" -f [guid]::NewGuid().ToString("N")
)
$remoteArchive = "/tmp/{0}" -f (Split-Path $archivePath -Leaf)
$target = "${UserName}@${OrangePiHost}"

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
    }
}

foreach ($commandName in @("git", "scp", "ssh")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $commandName"
    }
}

Push-Location $projectRoot
try {
    $pending = & git status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect the Git working tree."
    }
    if ($pending) {
        throw "Commit or stash local changes before deployment."
    }

    Invoke-External {
        & git archive --format=tar --output=$archivePath HEAD
    } "Unable to create the Git archive"

    Write-Host "Uploading committed project files to $target ..."
    Invoke-External {
        & scp $archivePath "${target}:$remoteArchive"
    } "Upload failed"

    $remoteCommand = "mkdir -p '$RemoteDirectory' && " +
        "tar -xf '$remoteArchive' -C '$RemoteDirectory' && " +
        "rm -f '$remoteArchive'"
    Invoke-External {
        & ssh $target $remoteCommand
    } "Remote extraction failed"

    Write-Host "Deployment completed: ${target}:$RemoteDirectory"
    Write-Host "Recordings, models, datasets, and virtual environments were not uploaded."
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
}
