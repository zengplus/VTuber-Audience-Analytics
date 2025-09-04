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

-- 6. 导出 CSV（可选）
COPY monthly_matrix   TO 'source_matrix.csv'  (HEADER, DELIMITER ',');
COPY monthly_matrix_out TO 'target_matrix.csv' (HEADER, DELIMITER ',');