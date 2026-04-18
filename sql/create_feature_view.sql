CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.avazu_feature` AS
WITH base AS (
    SELECT
        id,
        click,
        hour,
        PARSE_TIMESTAMP('%y%m%d%H', hour) AS event_ts,
        C1,
        banner_pos,
        site_id,
        site_domain,
        site_category,
        app_id,
        app_domain,
        app_category,
        device_id,
        device_ip,
        device_model,
        device_type,
        device_conn_type,
        C14,
        C15,
        C16,
        C17,
        C18,
        C19,
        C20,
        C21,
    FROM `${PROJECT_ID}.${DATASET_ID}.avazu_raw`
)
SELECT
    id,
    click,
    event_ts,
    EXTRACT(HOUR FROM event_ts) AS hour_of_day,
    EXTRACT(DAYOFWEEK FROM event_ts) AS day_of_week,
    CASE
        WHEN EXTRACT(DAYOFWEEK FROM event_ts) IN (1, 7) THEN 1
        ELSE 0
    END AS is_weekend,
    C1,
    banner_pos,
    site_id,
    site_domain,
    site_category,
    app_id,
    app_domain,
    app_category,
    device_id,
    device_ip,
    device_model,
    device_type,
    device_conn_type,
    C14,
    C15,
    C16,
    C17,
    C18,
    C19,
    C20,
    C21
FROM base;
    