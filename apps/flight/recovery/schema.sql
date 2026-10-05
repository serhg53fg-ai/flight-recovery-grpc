CREATE TABLE IF NOT EXISTS recovery_schema (
    singleton TINYINT PRIMARY KEY,
    version INT NOT NULL,
    CHECK (singleton = 1)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS recovery_jobs (
    recovery_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    submission_key VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    request_digest CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    created_at VARCHAR(40) CHARACTER SET ascii NOT NULL,
    source_job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    document JSON NOT NULL,
    UNIQUE KEY recovery_submission (submission_key),
    INDEX recovery_created (created_at, recovery_id),
    INDEX recovery_source (source_job_id, created_at)
) ENGINE=InnoDB;
