# ============================================================
# ELECTRICITY CONSUMPTION ANALYTICS
# DEMO RUNNER
#
# Chức năng:
# 1. Kiểm tra Python environment
# 2. Đọc InfluxDB token
# 3. Kiểm tra InfluxDB
# 4. Tự khởi động InfluxDB nếu chưa chạy
# 5. Chạy Streamlit dashboard
# ============================================================


# ============================================================
# 1. PROJECT ROOT
# ============================================================

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path


# ============================================================
# 2. ĐƯỜNG DẪN
# ============================================================

$PythonExe = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\python.exe"

$InfluxExe = Join-Path `
    $ProjectRoot `
    "influxdb\influxdb3.exe"

$InfluxDataDir = Join-Path `
    $ProjectRoot `
    "influxdb-data"

$TokenFile = Join-Path `
    $ProjectRoot `
    "influxdb-token.txt"

$DashboardFile = Join-Path `
    $ProjectRoot `
    "dashboard.py"


# ============================================================
# 3. KIỂM TRA FILE CẦN THIẾT
# ============================================================

Write-Host ""
Write-Host "=============================================="
Write-Host " Electricity Consumption Analytics - DEMO"
Write-Host "=============================================="
Write-Host ""


if (-not (Test-Path $PythonExe)) {

    Write-Host "ERROR: Python environment not found:"
    Write-Host $PythonExe

    exit 1
}


if (-not (Test-Path $InfluxExe)) {

    Write-Host "ERROR: InfluxDB executable not found:"
    Write-Host $InfluxExe

    exit 1
}


if (-not (Test-Path $TokenFile)) {

    Write-Host "ERROR: InfluxDB token file not found:"
    Write-Host $TokenFile

    exit 1
}


if (-not (Test-Path $DashboardFile)) {

    Write-Host "ERROR: dashboard.py not found:"
    Write-Host $DashboardFile

    exit 1
}


Write-Host "[OK] Project files found."


# ============================================================
# 4. NẠP TOKEN
# ============================================================

$env:INFLUXDB3_AUTH_TOKEN = (
    Get-Content $TokenFile -Raw
).Trim()


if (-not $env:INFLUXDB3_AUTH_TOKEN) {

    Write-Host "ERROR: InfluxDB token is empty."

    exit 1
}


Write-Host "[OK] InfluxDB token loaded."


# ============================================================
# 5. KIỂM TRA INFLUXDB PORT
# ============================================================

$InfluxRunning = Test-NetConnection `
    -ComputerName "127.0.0.1" `
    -Port 8181 `
    -InformationLevel Quiet


# ============================================================
# 6. KHỞI ĐỘNG INFLUXDB NẾU CHƯA CHẠY
# ============================================================

if (-not $InfluxRunning) {

    Write-Host ""
    Write-Host "[INFO] InfluxDB is not running."
    Write-Host "[INFO] Starting InfluxDB..."


    $InfluxArguments = @(
        "serve",
        "--node-id",
        "electricity-node",
        "--object-store",
        "file",
        "--data-dir",
        $InfluxDataDir,
        "--http-bind",
        "127.0.0.1:8181"
    )


    Start-Process `
        -FilePath $InfluxExe `
        -ArgumentList $InfluxArguments `
        -WorkingDirectory $ProjectRoot


    # --------------------------------------------------------
    # Chờ InfluxDB khởi động
    # --------------------------------------------------------

    $MaxAttempts = 20

    $Attempt = 0

    do {

        Start-Sleep -Seconds 1

        $Attempt++


        $InfluxRunning = Test-NetConnection `
            -ComputerName "127.0.0.1" `
            -Port 8181 `
            -InformationLevel Quiet


        if ($InfluxRunning) {

            Write-Host "[OK] InfluxDB started."

            break
        }


        Write-Host (
            "[INFO] Waiting for InfluxDB... {0}/{1}" `
            -f $Attempt,
            $MaxAttempts
        )


    } while (
        $Attempt -lt $MaxAttempts
    )


    if (-not $InfluxRunning) {

        Write-Host ""
        Write-Host "ERROR: InfluxDB failed to start."

        exit 1
    }

}
else {

    Write-Host "[OK] InfluxDB is already running."
}


# ============================================================
# 7. KIỂM TRA HEALTH
# ============================================================

Write-Host ""
Write-Host "[INFO] Checking InfluxDB health..."


try {

    $HealthResponse = Invoke-WebRequest `
        -Uri "http://127.0.0.1:8181/health" `
        -Headers @{
            Authorization = "Bearer $env:INFLUXDB3_AUTH_TOKEN"
        } `
        -UseBasicParsing `
        -ErrorAction Stop


    if ($HealthResponse.StatusCode -eq 200) {

        Write-Host "[OK] InfluxDB health check passed."

    }
    else {

        Write-Host "ERROR: InfluxDB health check failed."

        exit 1
    }

}
catch {

    Write-Host ""
    Write-Host "ERROR: Cannot connect to InfluxDB."

    Write-Host $_.Exception.Message

    exit 1
}


# ============================================================
# 8. KIỂM TRA PYTHON
# ============================================================

Write-Host ""
Write-Host "[INFO] Checking Python environment..."


& $PythonExe -c "import pyspark, pandas, pyarrow, streamlit, plotly, influxdb_client_3; print('Python environment: OK'); print('PySpark:', pyspark.__version__)"


if ($LASTEXITCODE -ne 0) {

    Write-Host ""
    Write-Host "ERROR: Python environment check failed."

    exit 1
}


# ============================================================
# 9. CHẠY STREAMLIT
# ============================================================

Write-Host ""
Write-Host "=============================================="
Write-Host " Starting Electricity Analytics Dashboard"
Write-Host "=============================================="
Write-Host ""

Write-Host "Dashboard:"
Write-Host "http://localhost:8501"

Write-Host ""
Write-Host "Press Ctrl+C to stop Streamlit."
Write-Host "InfluxDB will continue running."
Write-Host ""


# ============================================================
# 10. STREAMLIT
# ============================================================

& $PythonExe `
    -m streamlit `
    run `
    $DashboardFile `
    --server.port 8501 `
    --server.address 127.0.0.1