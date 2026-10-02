CREATE TABLE IF NOT EXISTS durable_schema (
    singleton TINYINT PRIMARY KEY,
    version INT NOT NULL,
    CHECK (singleton=1)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS queue_admission (
    singleton TINYINT PRIMARY KEY,
    capacity INT UNSIGNED NOT NULL,
    CHECK (singleton=1), CHECK (capacity>0)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS durable_jobs (
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    submission_key VARCHAR(128) COLLATE utf8mb4_bin NOT NULL UNIQUE,
    input_digest CHAR(64) CHARACTER SET ascii NOT NULL,
    deadline_at DATETIME(6) NOT NULL,
    event_seq BIGINT UNSIGNED NOT NULL DEFAULT 0,
    document JSON NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS durable_flights (
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    flight_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    ordinal INT UNSIGNED NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii NOT NULL,
    fence INT UNSIGNED NOT NULL DEFAULT 0,
    owner VARCHAR(128) COLLATE utf8mb4_bin NULL,
    trace_id CHAR(36) CHARACTER SET ascii NULL,
    lease_until DATETIME(6) NULL,
    input_document JSON NOT NULL,
    result_document JSON NULL,
    UNIQUE KEY flight_order (job_id,ordinal),
    INDEX flight_recovery (status,lease_until),
    FOREIGN KEY (job_id) REFERENCES durable_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS durable_attempts (
    flight_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    fence INT UNSIGNED NOT NULL,
    trace_id CHAR(36) CHARACTER SET ascii NOT NULL UNIQUE,
    owner VARCHAR(128) COLLATE utf8mb4_bin NOT NULL,
    status VARCHAR(16) CHARACTER SET ascii NOT NULL,
    started_at DATETIME(6) NOT NULL,
    finished_at DATETIME(6) NULL,
    PRIMARY KEY (flight_id,fence),
    FOREIGN KEY (flight_id) REFERENCES durable_flights(flight_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE TABLE IF NOT EXISTS durable_events (
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    event_id BIGINT UNSIGNED NOT NULL,
    document JSON NOT NULL,
    PRIMARY KEY (job_id,event_id),
    FOREIGN KEY (job_id) REFERENCES durable_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS prediction_outbox (
    message_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin PRIMARY KEY,
    job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    flight_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
    published_at DATETIME(6) NULL,
    failures INT UNSIGNED NOT NULL DEFAULT 0,
    next_publish_at DATETIME(6) NOT NULL,
    INDEX outbox_due (published_at,next_publish_at),
    FOREIGN KEY (flight_id) REFERENCES durable_flights(flight_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES durable_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB;
