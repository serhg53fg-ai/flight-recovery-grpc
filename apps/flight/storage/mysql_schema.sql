CREATE TABLE IF NOT EXISTS storage_schema (
    singleton TINYINT PRIMARY KEY,
    version INT NOT NULL,
    CHECK (singleton = 1)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS prediction_jobs (
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    created_at VARCHAR(40) CHARACTER SET ascii NOT NULL,
    document JSON NOT NULL,
    INDEX jobs_created (created_at, job_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS flight_predictions (
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    ordinal INT UNSIGNED NOT NULL,
    flight_id VARCHAR(128) COLLATE utf8mb4_bin NULL,
    document JSON NOT NULL,
    PRIMARY KEY (job_id, ordinal),
    UNIQUE KEY flight_identity (job_id, flight_id),
    FOREIGN KEY (job_id) REFERENCES prediction_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
