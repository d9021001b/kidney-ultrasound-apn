$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path

$buckets = @{
    submission = Join-Path $root "01_submission"
    experiments = Join-Path $root "02_experiments"
    data = Join-Path $root "03_data_models"
    archive = Join-Path $root "04_quality_archive"
    code = Join-Path $root "05_code"
}
foreach ($path in $buckets.Values) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $path "root_files") -Force | Out-Null
}

function Move-DirectoryCompat([string]$name, [string]$bucket) {
    $old = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $old -PathType Container)) { return }
    $item = Get-Item -LiteralPath $old -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return }
    $new = Join-Path $buckets[$bucket] $name
    if (Test-Path -LiteralPath $new -PathType Container) {
        # A previous run may have moved most children before a preview lock.
        Get-ChildItem -LiteralPath $old -Force | ForEach-Object {
            $childTarget = Join-Path $new $_.Name
            if (Test-Path -LiteralPath $childTarget) { return }
            try { Move-Item -LiteralPath $_.FullName -Destination $new }
            catch { Write-Warning "Skipped locked child: $($_.FullName)" }
        }
        if ((Get-ChildItem -LiteralPath $old -Force | Measure-Object).Count -eq 0) {
            Remove-Item -LiteralPath $old
        } else {
            Write-Warning "Directory remains partially locked: $old"
        }
        return
    }
    try {
        Move-Item -LiteralPath $old -Destination $new
    } catch {
        Write-Warning "Skipped locked directory: $old"
        return
    }
}

function Move-FileCompat([IO.FileInfo]$file, [string]$bucket) {
    $old = $file.FullName
    if ($file.Name -eq "organize_project_layout.ps1") { return }
    $targetDir = Join-Path $buckets[$bucket] "root_files"
    $new = Join-Path $targetDir $file.Name
    if (Test-Path -LiteralPath $new) { throw "Destination already exists: $new" }
    Move-Item -LiteralPath $old -Destination $new
}

$submissionDirs = @(
    "CYCH_BMC_Medical_Imaging_submission_draft_20260915",
    "CYCH_journal_submission_package_v2_20260914"
)
$dataDirs = @(
    "CYCH", "NCKUH", "cych_0", "cych_0_1", "cych_0_1_inner", "cych_1",
    "cych_bg_black", "cych_bg_white", "cych_bw", "cych_heatmap", "cych_heatmap2",
    "cych_image_label", "nckuh_0", "nckuh_0_1", "nckuh_0_1_inner", "nckuh_1",
    "nckuh_bg_black", "nckuh_bg_white", "nckuh_bw", "nckuh_heatmap", "nckuh_heatmap2",
    "nckuh_image_label", "original_png", "original_png_nckuh", "ki-data", "yolo_crops",
    "kidney_label_request", "kidney_labels", "partitions", "patient_characteristics",
    "external_models", "dinov3_vith16plus_weights_20260913", "third_party"
)
$archiveDirs = @(
    "__pycache__", ".pytest_cache", ".tmp_chrome_history", "augmentation_20260912",
    "docx_read_original_0307a_qa", "journal_figures_600dpi", "Kidney_Inflammation_Deliverables_20260907",
    "project_audit_20260912", "reproduction_20260912", "results", "results_controls",
    "results_feasibility", "results_green", "results_pooled", "results_single_center", "results_yolo",
    "review_20260909", "scratch_kidata", "scratch_transfer", "segmented_protocol_80_20", "tmp"
)

foreach ($name in $submissionDirs) { Move-DirectoryCompat $name "submission" }
foreach ($name in $dataDirs) { Move-DirectoryCompat $name "data" }
foreach ($name in $archiveDirs) { Move-DirectoryCompat $name "archive" }

# Every dated CYCH directory is an experiment unless classified above.
Get-ChildItem -LiteralPath $root -Directory -Force |
    Where-Object { $_.Name -like "CYCH_*" } |
    ForEach-Object { Move-DirectoryCompat $_.Name "experiments" }

# Remaining QA/render directories are derived artifacts and are kept together.
Get-ChildItem -LiteralPath $root -Directory -Force |
    Where-Object { $_.Name -like "docx_qa_*" -or $_.Name -like "qa_render_*" } |
    ForEach-Object { Move-DirectoryCompat $_.Name "archive" }

# Keep the hidden virtual environment and Serena state in place. They are local
# tooling state rather than project evidence and are already hidden in Explorer.
$codeExtensions = @(".py", ".ps1", ".js", ".mjs", ".bat", ".cmd")
$modelExtensions = @(".pt", ".pth", ".npy", ".npz", ".onnx", ".safetensors", ".tar", ".gz")
$submissionExtensions = @(".docx", ".pdf")
Get-ChildItem -LiteralPath $root -File -Force | ForEach-Object {
    if ($_.Name -eq "organize_project_layout.ps1" -or $_.Name -eq "PROJECT_LAYOUT_20260915.md" -or (($_.Attributes -band [IO.FileAttributes]::Hidden) -ne 0)) { return }
    if ($codeExtensions -contains $_.Extension.ToLowerInvariant()) {
        Move-FileCompat $_ "code"
    } elseif ($modelExtensions -contains $_.Extension.ToLowerInvariant() -or $_.Name -eq "weights.tar.gz") {
        Move-FileCompat $_ "data"
    } elseif ($submissionExtensions -contains $_.Extension.ToLowerInvariant()) {
        Move-FileCompat $_ "submission"
    } elseif ($_.Extension.ToLowerInvariant() -eq ".zip" -and ($_.Name -like "CYCH_*" -or $_.Name -like "Kidney_Inflammation*")) {
        Move-FileCompat $_ "submission"
    } else {
        Move-FileCompat $_ "archive"
    }
}

$layout = @(
    "# Project layout",
    "",
    "The project is organized into five canonical areas:",
    "",
    "- 01_submission: latest manuscripts, cover letter, supplementary files, and submission archives.",
    "- 02_experiments: dated model, feature, ablation, segmentation-QC, and sensitivity runs.",
    "- 03_data_models: source images, labels, patient tables, pretrained weights, and external models.",
    "- 04_quality_archive: rendered DOCX QA, old deliverables, scratch outputs, logs, and derived reports.",
    "- 05_code: analysis scripts and reproducibility utilities.",
    "",
    "graphify-out remains at the project root because Graphify uses that location as its persistent index.",
    "",
    "Former root-level aliases have been consolidated under 04_quality_archive/legacy_root_aliases. They are retained for traceability, not as active project entry points. New work should use the canonical folders above.",
    "",
    "Latest manuscript:",
    "01_submission/root_files/yolo_aug_triEMR_PCA_SVM_renal_2022_v2_journal_ready.docx",
    "01_submission/CYCH_BMC_Medical_Imaging_submission_draft_20260915/01_Main_manuscript_BMC_Medical_Imaging_draft.docx"
) -join [Environment]::NewLine
$layoutPath = Join-Path $root "PROJECT_LAYOUT_20260915.md"
$layout | Set-Content -LiteralPath $layoutPath -Encoding UTF8
Write-Output "Project layout organized. Canonical folders: $($buckets.Values -join ', ')"
