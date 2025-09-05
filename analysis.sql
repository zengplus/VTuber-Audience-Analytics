-- 0. 建库（如已存在可跳过）
-- duckdb mydb.duckdb < analysis.sql

-- 0.1 把主播维度一次性写进去（仅第一次跑时生效）
CREATE TABLE IF NOT EXISTS dim_liver (
    liver BIGINT PRIMARY KEY,
    name  VARCHAR
);

-- 把你已有的 40+ 位主播 + YLG 全量插入
INSERT OR REPLACE INTO dim_liver VALUES
(672328094,'嘉然'),
(672346917,'向晚'),
(672353429,'贝拉'),
(672342685,'乃琳'),
(351609538,'珈乐'),
(3537115310721181,'心宜'),
(3537115310721781,'思诺'),
(440738032,'安可'),
(698438232,'扇宝'),
(401315430,'星瞳'),
(1660392980,'恬豆'),
(1217754423,'又一'),
(1878154667,'沐霂'),
(1900141897,'梨安'),
(1778026586,'米诺'),
(1875044092,'莞儿'),
(1811071010,'虞莫'),
(1669777785,'露早'),
(1795147802,'柚恩'),
(7706705,'阿梓'),
(434334701,'七海'),
(666726799,'悠亚'),
(480680646,'阿萨'),
(14387072,'小可'),
(477317922,'弥希'),
(1265680561,'塔菲'),
(2132180406,'奶绿'),
(1265605287,'麻尤米'),
(1908273021,'露娜'),
(1855519979,'dodo'),
(2000609327,'露米'),
(1377212676,'永恒娘'),
(686201061,'古堡龙姬'),
(3461582034045213,'宣小纸'),
(1011797664,'卡缇娅'),
(3461578781362279,'叶河黎'),
(1383815813,'吉诺儿'),
(1219196749,'唐九夏'),
(1734978373,'小柔'),
(1501380958,'艾露露'),
(3493139945884106,'雪糕'),
(51030552,'星汐'),
(15641218,'笙歌'),
(3821157,'东爱璃'),
(-1,'YLG');

-- 1. 映射事件表
CREATE OR REPLACE TABLE events AS
SELECT uid,
       liver,
       ts,
       DATE_TRUNC('month', ts) AS month
-- 从 Parquet 文件导入
FROM 'fans_events.parquet';

-- 2. 每个主播每月首次关注 = 新增观众
CREATE OR REPLACE TABLE liver_new AS
SELECT liver,
       month,
       uid
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY uid, liver ORDER BY ts) AS rn
    FROM events
) t
WHERE rn = 1;

-- 3. 观众在关注某主播之前，最常看的主播（排除自己）
CREATE OR REPLACE TABLE new_source AS
SELECT n.liver  AS target_liver,
       n.month,
       COALESCE(
           (SELECT e2.liver
            FROM events e2
            WHERE e2.uid = n.uid
              AND e2.liver <> n.liver
              AND e2.ts < n.month
            GROUP BY e2.liver
            ORDER BY COUNT(*) DESC
            LIMIT 1),
       -1) AS source_liver
FROM liver_new n;

-- 4. 观众在关注某主播之后，最常看的主播（排除自己）
CREATE OR REPLACE TABLE new_target AS
SELECT n.liver  AS source_liver,
       n.month,
       COALESCE(
           (SELECT e2.liver
            FROM events e2
            WHERE e2.uid = n.uid
              AND e2.liver <> n.liver
              AND e2.ts >= n.month + INTERVAL 1 month
            GROUP BY e2.liver
            ORDER BY COUNT(*) DESC
            LIMIT 1),
       -1) AS target_liver
FROM liver_new n;

-- 5. 预计算月度矩阵
CREATE OR REPLACE TABLE monthly_matrix AS
SELECT target_liver,
       DATE_TRUNC('month', month) AS month,
       source_liver,
       COUNT(*) AS cnt
FROM new_source
GROUP BY ALL;

CREATE OR REPLACE TABLE monthly_matrix_out AS
SELECT source_liver,
       DATE_TRUNC('month', month) AS month,
       target_liver,
       COUNT(*) AS cnt
FROM new_target
GROUP BY ALL;

-- 6. AARRRR 漏斗指标（主播×月度）
CREATE OR REPLACE TABLE aarr_metrics AS
WITH m0 AS (
    /* Acquisition：本月新增关注数 */
    SELECT liver,
           month,
           COUNT(*) AS acq
    FROM liver_new
    GROUP BY ALL
),
m1 AS (
    /* Activation：新增后 7 日内再次观看的比例 */
    SELECT n.liver,
           n.month,
           COUNT(DISTINCT n.uid) FILTER (
               WHERE EXISTS (
                   SELECT 1
                   FROM events e
                   WHERE e.uid = n.uid
                     AND e.liver = n.liver
                     AND e.ts BETWEEN n.month AND n.month + INTERVAL 7 DAY
               )
           ) AS activ,
           COUNT(DISTINCT n.uid) AS acq2
    FROM liver_new n
    GROUP BY ALL
),
m2 AS (
    /* Retention：新增后第 30±3 天仍观看的人数 */
    SELECT n.liver,
           n.month,
           COUNT(DISTINCT n.uid) FILTER (
               WHERE EXISTS (
                   SELECT 1
                   FROM events e
                   WHERE e.uid = n.uid
                     AND e.liver = n.liver
                     AND e.ts BETWEEN n.month + INTERVAL 27 DAY
                                  AND n.month + INTERVAL 33 DAY
               )
           ) AS reten
    FROM liver_new n
    GROUP BY ALL
),
m3 AS (
    /* Referral：新增用户里带来≥1 个“流入”的人数（source=该主播） */
    SELECT s.target_liver AS liver,
           s.month,
           COUNT(DISTINCT s.uid) AS refer
    FROM new_source s
    WHERE s.source_liver = s.target_liver   -- 自引
    GROUP BY ALL
),
m4 AS (
    /* Revenue（这里用“观看次数”代充）：
       新增用户在随后 30 天内的总观看次数 */
    SELECT n.liver,
           n.month,
           SUM(e.cnt) AS rev
    FROM liver_new n
    JOIN (
        SELECT uid,liver,COUNT(*) cnt,MIN(ts) AS fst
        FROM events
        GROUP BY ALL
    ) e
      ON e.uid = n.uid AND e.liver = n.liver
    WHERE e.fst BETWEEN n.month AND n.month + INTERVAL 30 DAY
    GROUP BY ALL
)
SELECT a.liver,
       a.month,
       a.acq,
       COALESCE(b.activ,0) AS activ,
       ROUND(COALESCE(b.activ,0)*1.0/NULLIF(a.acq,0),3) AS activ_rate,
       COALESCE(c.reten,0) AS reten,
       ROUND(COALESCE(c.reten,0)*1.0/NULLIF(a.acq,0),3) AS reten_rate,
       COALESCE(d.refer,0) AS refer,
       ROUND(COALESCE(d.refer,0)*1.0/NULLIF(a.acq,0),3) AS refer_rate,
       COALESCE(e.rev,0) AS revenue
INTO aarr_metrics
FROM m0 a
LEFT JOIN m1 b USING(liver,month)
LEFT JOIN m2 c USING(liver,month)
LEFT JOIN m3 d USING(liver,month)
LEFT JOIN m4 e USING(liver,month)
ORDER BY a.month, a.liver;

-- 7. RFM 用户分值（近90 天窗口）
CREATE OR REPLACE TABLE rfm_user AS
WITH base AS (
    SELECT uid,
           liver,
           DATE_DIFF('day', MAX(ts), CURRENT_DATE) AS R,   -- Recency
           COUNT(*)                                     AS F,   -- Frequency
           COUNT(*)                                     AS M    -- 无金额→用次数代
    FROM events
    WHERE ts >= CURRENT_DATE - INTERVAL 90 DAY
    GROUP BY uid, liver
),
qt AS (
    SELECT liver,
           PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY R) AS r_low,
           PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY R) AS r_high,
           PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY F) AS f_low,
           PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY F) AS f_high,
           PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY M) AS m_low,
           PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY M) AS m_high
    FROM base
    GROUP BY liver
),
score AS (
    SELECT b.*,
           qt.r_low, qt.r_high, qt.f_low, qt.f_high, qt.m_low, qt.m_high,
           CASE WHEN b.R <= qt.r_high THEN 5
                WHEN b.R <= qt.r_low  THEN 1
                ELSE 3 END AS r_score,
           CASE WHEN b.F >= qt.f_high THEN 5
                WHEN b.F >= qt.f_low  THEN 3
                ELSE 1 END AS f_score,
           CASE WHEN b.M >= qt.m_high THEN 5
                WHEN b.M >= qt.m_low  THEN 3
                ELSE 1 END AS m_score
    FROM base b
    JOIN qt USING(liver)
)
SELECT uid,
       liver,
       R, F, M,
       r_score, f_score, m_score,
       CAST(r_score AS VARCHAR) || CAST(f_score AS VARCHAR) || CAST(m_score AS VARCHAR) AS rfm,
       CASE
         WHEN r_score>=4 AND f_score>=4 AND m_score>=4 THEN '高价值忠诚'
         WHEN r_score>=3 AND f_score>=3 THEN '潜力用户'
         WHEN r_score<=2 THEN '流失风险'
         ELSE '一般用户'
       END AS rfm_label
INTO rfm_user
FROM score
ORDER BY liver, r_score DESC, f_score DESC, m_score DESC;
