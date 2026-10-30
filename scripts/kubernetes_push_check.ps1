param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("master", "chakri")]
    [string]$Branch,

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{40}$")]
    [string]$Commit
)

$ErrorActionPreference = "Stop"
$repository = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not locate the Git repository."
}
Set-Location -LiteralPath $repository

$currentBranch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or $currentBranch -ne $Branch) {
    throw "Check out local branch '$Branch' before pushing it."
}

$head = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head -ne $Commit) {
    throw "The pushed commit must be the current HEAD."
}

$workingChanges = & git status --porcelain --untracked-files=normal
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Git working tree."
}
if ($workingChanges) {
    throw "Commit or remove tracked and untracked changes before pushing."
}

foreach ($command in @("python", "docker", "kubectl")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command '$command' was not found on PATH."
    }
}

Write-Host "Running tests for $Branch at $Commit..."
& python -m pytest
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed."
}

& python -m ruff check .
if ($LASTEXITCODE -ne 0) {
    throw "Ruff checks failed."
}

$context = (& kubectl config current-context).Trim()
if ($LASTEXITCODE -ne 0 -or $context -ne "docker-desktop") {
    throw "The active Kubernetes context must be 'docker-desktop'."
}

& kubectl get deployment/readmission-risk-api -n default
if ($LASTEXITCODE -ne 0) {
    throw "The readmission API deployment is not available in Kubernetes."
}

$containerRunning = (& docker inspect --format "{{.State.Running}}" desktop-control-plane).Trim()
if ($LASTEXITCODE -ne 0 -or $containerRunning -ne "true") {
    throw "The Docker Desktop Kubernetes control-plane container was not found or is not running."
}

$image = "readmission-risk-api:pre-push-$($Commit.Substring(0, 12))"
$archive = Join-Path $env:TEMP "readmission-risk-api-$Commit.tar"
$archiveCopied = $false

try {
    Write-Host "Building $image..."
    & docker build --tag $image .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed."
    }

    & docker save --output $archive $image
    if ($LASTEXITCODE -ne 0) {
        throw "Could not export the Docker image."
    }

    & docker cp $archive desktop-control-plane:/root/readmission-risk-api.tar
    if ($LASTEXITCODE -ne 0) {
        throw "Could not copy the image to the Kubernetes control plane."
    }
    $archiveCopied = $true

    & docker exec desktop-control-plane ctr -n k8s.io images import /root/readmission-risk-api.tar
    if ($LASTEXITCODE -ne 0) {
        throw "Could not import the image into Kubernetes."
    }

    Write-Host "Deploying and waiting for Kubernetes rollout..."
    & python deploy_with_rollback.py $image --timeout 300s
    if ($LASTEXITCODE -ne 0) {
        throw "Kubernetes deployment failed; the deployment script attempted rollback."
    }

    $healthCheck = "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10); body = json.load(response); assert response.status == 200 and body.get('status') == 'ok', body; print('Kubernetes API health check passed.')"
    & kubectl exec deployment/readmission-risk-api -n default -- python -c $healthCheck
    if ($LASTEXITCODE -ne 0) {
        throw "The deployed API health check failed."
    }

    Write-Host "All pre-push checks passed for $Branch at $Commit."
}
finally {
    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    if ($archiveCopied) {
        & docker exec desktop-control-plane rm -f /root/readmission-risk-api.tar
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Could not remove the temporary image archive from the control plane."
        }
    }
}
