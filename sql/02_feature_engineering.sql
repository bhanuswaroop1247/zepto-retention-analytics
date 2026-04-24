-- ============================================================
-- Phase 2: Feature Engineering
-- Project Zepto — Delivery Delta Forecasting & Retention
-- Tables created in dependency order:
--   delivery_features → customer_features → customer_clv → analytical_mart
-- Note: [order] and [transaction] are wrapped in square brackets
--       because they are reserved words in SQLite.
-- ============================================================


-- ── 1. delivery_features ────────────────────────────────────
-- Adds delivery_delta (mins over 10-min target) and is_late flag
DROP TABLE IF EXISTS delivery_features;
CREATE TABLE delivery_features AS
SELECT
    d.order_id,
    d.delivery_time_mins,
    d.delivery_time_mins - 10                              AS delivery_delta,
    CASE WHEN d.delivery_time_mins > 10 THEN 1 ELSE 0 END AS is_late
FROM delivery d;


-- ── 2. customer_features ────────────────────────────────────
-- Aggregates per customer: RFM metrics, delivery stats, ratings
DROP TABLE IF EXISTS customer_features;
CREATE TABLE customer_features AS
SELECT
    c.customer_id,
    c.age,
    c.city,
    c.state,
    CAST(julianday('2024-12-31') - julianday(c.created_date) AS INTEGER)
                                                           AS days_since_registration,
    COUNT(DISTINCT o.order_id)                             AS total_orders,
    MIN(o.order_date)                                      AS first_order_date,
    MAX(o.order_date)                                      AS last_order_date,
    CAST(julianday('2024-12-31') - julianday(MAX(o.order_date)) AS INTEGER)
                                                           AS days_since_last_order,
    CASE
        WHEN CAST(julianday('2024-12-31') - julianday(MAX(o.order_date)) AS INTEGER) > 90
        THEN 1 ELSE 0
    END                                                    AS churn_flag,
    AVG(df.delivery_delta)                                 AS avg_delivery_delta,
    MAX(df.delivery_delta)                                 AS max_delivery_delta,
    AVG(df.is_late) * 100.0                                AS pct_late_orders,
    AVG(r.rating)                                          AS avg_rating,
    MIN(r.rating)                                          AS min_rating,
    SUM(CASE WHEN o.order_status IN ('Cancelled', 'Returned') THEN 1 ELSE 0 END)
        * 100.0 / COUNT(DISTINCT o.order_id)               AS pct_cancelled_returned
FROM customer c
LEFT JOIN [order]          o  ON c.customer_id = o.customer_id
LEFT JOIN delivery_features df ON o.order_id   = df.order_id
LEFT JOIN rating            r  ON o.order_id   = r.order_id
GROUP BY c.customer_id, c.age, c.city, c.state;


-- ── 3. customer_clv ─────────────────────────────────────────
-- Sums transaction amounts per customer (joined via [order])
DROP TABLE IF EXISTS customer_clv;
CREATE TABLE customer_clv AS
SELECT
    o.customer_id,
    SUM(t.amount) AS total_clv
FROM [transaction] t
JOIN [order] o ON t.order_id = o.order_id
GROUP BY o.customer_id;


-- ── 4. analytical_mart ──────────────────────────────────────
-- Master table: customer_features + CLV tier
-- Filtered to customers with at least 1 order (total_orders > 0)
DROP TABLE IF EXISTS analytical_mart;
CREATE TABLE analytical_mart AS
SELECT
    cf.*,
    COALESCE(clv.total_clv, 0) AS total_clv,
    CASE
        WHEN COALESCE(clv.total_clv, 0) <= 1840 THEN 'Low'
        WHEN COALESCE(clv.total_clv, 0) <= 6030 THEN 'Mid'
        ELSE 'High'
    END AS clv_tier
FROM customer_features cf
LEFT JOIN customer_clv clv ON cf.customer_id = clv.customer_id
WHERE cf.total_orders > 0;
