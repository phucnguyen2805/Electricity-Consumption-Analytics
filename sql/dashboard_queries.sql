-- ============================================================
-- ELECTRICITY FORECAST DASHBOARD QUERIES
-- ============================================================


-- ============================================================
-- 1. COUNT POINTS
-- ============================================================

SELECT COUNT(*) AS total_points
FROM electricity_forecast;


-- ============================================================
-- 2. COUNT SITE
-- ============================================================

SELECT COUNT(DISTINCT site_id) AS sites
FROM electricity_forecast;


-- ============================================================
-- 3. HOURLY
-- ============================================================

SELECT
    DATE_PART('hour', time) AS hour_of_day,
    AVG(actual) AS actual_consumption,
    AVG(prediction) AS predicted_consumption,
    AVG(absolute_error) AS MAE
FROM electricity_forecast
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- 4. DAILY
-- ============================================================

SELECT
    DATE_TRUNC('day', time) AS date,
    SUM(actual) AS actual_consumption,
    SUM(prediction) AS predicted_consumption,
    AVG(absolute_error) AS MAE
FROM electricity_forecast
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- 5. BY SITE
-- ============================================================

SELECT
    site_id,
    SUM(actual) AS actual_consumption,
    SUM(prediction) AS predicted_consumption,
    AVG(absolute_error) AS MAE,
    COUNT(*) AS records
FROM electricity_forecast
GROUP BY site_id
ORDER BY actual_consumption DESC;


-- ============================================================
-- 6. ACTUAL VS PREDICTED
-- ============================================================

SELECT
    time,
    site_id,
    actual,
    prediction,
    absolute_error
FROM electricity_forecast
ORDER BY time;


-- ============================================================
-- 7. TOP 20 ERRORS
-- ============================================================

SELECT
    time,
    site_id,
    actual,
    prediction,
    absolute_error,
    is_outlier
FROM electricity_forecast
ORDER BY absolute_error DESC
LIMIT 20;