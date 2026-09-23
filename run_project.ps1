<#
    CottonCross pipeline driver.

    .\run_project.ps1 -Mode tests                     run the unit tests
    .\run_project.ps1 -Mode prepare -Pdf <path>       rebuild the corpus from the PDF
    .\run_project.ps1 -Mode train                     the full experiment matrix (GPU, hours)
    .\run_project.ps1 -Mode report                    tables + figures from existing results
    .\run_project.ps1 -Mode review                    the offline inspection page
    .\run_project.ps1 -Mode full -Pdf <path>          everything, in order
#>
param(
    [ValidateSet('count','tests','prepare','train','counting','report','review','full')][string]$Mode = 'report',
    [string]$Input = '',
    [string]$Out = 'outputs/count',
    [string]$Pdf = '',
    [int]$Steps = 3000,
    [int]$Warmup = 2000,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Step {
    param([string[]]$Arguments)
    Write-Host ">> $Python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "step failed: $($Arguments -join ' ')" }
}

if ($Mode -eq 'count') {
    if (-not $Input) { throw 'pass -Input <image or folder>' }
    Invoke-Step @('-m','cottoncross.app','--input',$Input,'--out',$Out)
    exit
}

if ($Mode -eq 'tests') {
    Invoke-Step @('-m','unittest','discover','-s','tests','-v')
    exit
}

if ($Mode -in @('prepare','full')) {
    if ($Pdf) { Invoke-Step @('-m','cottoncross.prepare','--pdf',$Pdf) }
    Invoke-Step @('scripts/prepare_data.py')
}

if ($Mode -in @('train','full')) {
    Invoke-Step @('scripts/run_experiments.py','--steps',"$Steps",'--warmup',"$Warmup")
}

if ($Mode -in @('counting','train','full')) {
    Invoke-Step @('scripts/count_features.py')
    Invoke-Step @('-m','cottoncross.countbench')
    Invoke-Step @('scripts/count_features.py','--dataset','bench')
    Invoke-Step @('scripts/count_fibers.py')
    Invoke-Step @('scripts/count_report.py')
}

if ($Mode -in @('report','counting','train','full')) {
    Invoke-Step @('scripts/make_tables.py')
    Invoke-Step @('scripts/make_report.py')
    Invoke-Step @('scripts/fill_readme.py')
    Invoke-Step @('-m','cottoncross.figures','--all')
    Invoke-Step @('-m','cottoncross.figures','--all','--lang','zh')
    Invoke-Step @('scripts/tools/package_project.py')
}

if ($Mode -in @('review','full')) {
    Invoke-Step @('-m','cottoncross.review')
    Write-Host 'Open results/real/review/index.html to inspect the plates.' -ForegroundColor Green
}
