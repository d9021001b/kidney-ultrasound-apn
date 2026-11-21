$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path

$archive = Join-Path $root "04_quality_archive\legacy_root_aliases"
$legacyDirs = Join-Path $archive "directories"
$legacyFiles = Join-Path $archive "files"
New-Item -ItemType Directory -Path $legacyDirs -Force | Out-Null
New-Item -ItemType Directory -Path $legacyFiles -Force | Out-Null

# The old root entries are only compatibility aliases created by the layout pass.
# Move them together so the root remains a clear entry point to the project.
$preserve = @(".serena", ".venvs")
$movedDirs = New-Object System.Collections.Generic.List[string]
$movedFiles = New-Object System.Collections.Generic.List[string]
$rootItems = @(Get-ChildItem -LiteralPath $root -Force)
foreach ($item in $rootItems) {
    if ($preserve -contains $item.Name) { continue }
    if (($item.Attributes -band [IO.FileAttributes]::Hidden) -eq 0) { continue }

    if ($item.PSIsContainer) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            Write-Warning "Skipped non-link hidden directory: $($item.FullName)"
            continue
        }
        Move-Item -LiteralPath $item.FullName -Destination $legacyDirs
        $movedDirs.Add($item.Name)
    } else {
        Move-Item -LiteralPath $item.FullName -Destination $legacyFiles
        $movedFiles.Add($item.Name)
    }
}

# Canonical project folders should be browsable. Keep nested repository/cache
# internals untouched; only remove inherited hidden flags at the area boundary.
$canonicalAreas = @(
    "01_submission", "02_experiments", "03_data_models",
    "04_quality_archive", "05_code"
)
foreach ($area in $canonicalAreas) {
    $areaPath = Join-Path $root $area
    Get-ChildItem -LiteralPath $areaPath -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if (($_.Attributes -band [IO.FileAttributes]::Hidden) -ne 0) {
            $_.Attributes = $_.Attributes -band (-bnot [IO.FileAttributes]::Hidden)
        }
    }
}
foreach ($rootFiles in ($canonicalAreas | ForEach-Object { Join-Path $root "$_\root_files" })) {
    Get-ChildItem -LiteralPath $rootFiles -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if (($_.Attributes -band [IO.FileAttributes]::Hidden) -ne 0) {
            $_.Attributes = $_.Attributes -band (-bnot [IO.FileAttributes]::Hidden)
        }
    }
}

# The archive is intentionally visible and self-describing.
Get-ChildItem -LiteralPath $archive -Force | ForEach-Object {
    if (($_.Attributes -band [IO.FileAttributes]::Hidden) -ne 0) {
        $_.Attributes = $_.Attributes -band (-bnot [IO.FileAttributes]::Hidden)
    }
}
foreach ($legacyContainer in @($legacyDirs, $legacyFiles)) {
    Get-ChildItem -LiteralPath $legacyContainer -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if (($_.Attributes -band [IO.FileAttributes]::Hidden) -ne 0) {
            $_.Attributes = $_.Attributes -band (-bnot [IO.FileAttributes]::Hidden)
        }
    }
}
$manifest = @(
    "# Retired root-level aliases",
    "",
    "These entries were generated as compatibility aliases during the first project reorganization.",
    "They are retained here for traceability; canonical data and results remain in the five numbered project areas.",
    "",
    "## Directories",
    ""
)
$manifest += ($movedDirs | Sort-Object | ForEach-Object { "- $_" })
$manifest += @("", "## Files", "")
$manifest += ($movedFiles | Sort-Object | ForEach-Object { "- $_" })
$manifestPath = Join-Path $archive "README.md"
if ($movedDirs.Count -gt 0 -or $movedFiles.Count -gt 0 -or -not (Test-Path -LiteralPath $manifestPath)) {
    $manifest | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}

Write-Output ("Consolidated {0} hidden root directories and {1} hidden root files under {2}" -f $movedDirs.Count, $movedFiles.Count, $archive)
