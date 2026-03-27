BEGIN;

-- ---------------------------------------------------------------------------
-- Regions
-- ---------------------------------------------------------------------------
CREATE TABLE regions (
    id          SERIAL PRIMARY KEY,
    code        VARCHAR(20)  NOT NULL UNIQUE,
    name        VARCHAR(100) NOT NULL,
    boundary    GEOMETRY(MultiPolygon, 4326),
    config_path VARCHAR(200),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Weather events (ingested from NOAA Storm Events, NWS alerts, etc.)
-- ---------------------------------------------------------------------------
CREATE TABLE weather_events (
    id              BIGSERIAL,
    event_time      TIMESTAMPTZ   NOT NULL,
    source          VARCHAR(20),
    event_type      VARCHAR(60),
    magnitude       FLOAT,
    magnitude_type  VARCHAR(10),
    location        GEOMETRY(Point, 4326),
    h3_index_res7   VARCHAR(15),
    h3_index_res9   VARCHAR(15),
    state_fips      VARCHAR(5),
    county_fips     VARCHAR(5),
    damage_property NUMERIC,
    damage_crops    NUMERIC,
    injuries        INT           DEFAULT 0,
    deaths          INT           DEFAULT 0,
    narrative       TEXT,
    episode_id      BIGINT,
    raw_data        JSONB,
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Outage observations (Eagle-I / utility feeds)
-- ---------------------------------------------------------------------------
CREATE TABLE outage_observations (
    id              BIGSERIAL,
    observed_at     TIMESTAMPTZ   NOT NULL,
    county_fips     VARCHAR(5),
    state_fips      VARCHAR(2),
    customers_out   INT,
    total_customers INT,
    outage_fraction FLOAT,
    h3_index_res7   VARCHAR(15),
    source          VARCHAR(20)   DEFAULT 'eagle_i',
    ingested_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Grid load and capacity snapshots
-- ---------------------------------------------------------------------------
CREATE TABLE grid_load (
    id                BIGSERIAL,
    recorded_at       TIMESTAMPTZ  NOT NULL,
    region_code       VARCHAR(20),
    load_mw           FLOAT,
    capacity_mw       FLOAT,
    reserve_margin_pct FLOAT,
    frequency_hz      FLOAT,
    source            VARCHAR(20),
    ingested_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Static / slowly-changing infrastructure attributes per H3 cell
-- ---------------------------------------------------------------------------
CREATE TABLE infrastructure (
    id                    SERIAL PRIMARY KEY,
    h3_index_res7         VARCHAR(15),
    county_fips           VARCHAR(5),
    transmission_line_km  FLOAT   DEFAULT 0,
    distribution_line_km  FLOAT   DEFAULT 0,
    substations_count     INT     DEFAULT 0,
    avg_line_age_years    FLOAT,
    vegetation_density    FLOAT,
    land_cover_type       VARCHAR(30),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Socioeconomic context per H3 cell
-- ---------------------------------------------------------------------------
CREATE TABLE socioeconomic (
    id                       SERIAL PRIMARY KEY,
    h3_index_res7            VARCHAR(15),
    county_fips              VARCHAR(5),
    population_density       FLOAT,
    median_income            FLOAT,
    critical_facilities_count INT    DEFAULT 0,
    housing_age_median       FLOAT,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Pre-computed feature vectors for ML training / serving
-- ---------------------------------------------------------------------------
CREATE TABLE feature_store (
    id                       BIGSERIAL,
    computed_at              TIMESTAMPTZ  NOT NULL,
    h3_index_res7            VARCHAR(15),
    feature_version          VARCHAR(10),
    features                 JSONB,
    target_outage_occurred   BOOLEAN,
    target_customers_affected INT,
    ingested_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Model predictions
-- ---------------------------------------------------------------------------
CREATE TABLE predictions (
    id                BIGSERIAL,
    predicted_at      TIMESTAMPTZ  NOT NULL,
    h3_index_res7     VARCHAR(15),
    region_code       VARCHAR(20),
    model_version     VARCHAR(50),
    risk_probability  FLOAT,
    uncertainty_lower FLOAT,
    uncertainty_upper FLOAT,
    risk_level        VARCHAR(10),
    features_snapshot JSONB,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Operational alerts
-- ---------------------------------------------------------------------------
CREATE TABLE alerts (
    id                 BIGSERIAL PRIMARY KEY,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    expires_at         TIMESTAMPTZ,
    region_code        VARCHAR(20),
    h3_index_res7      VARCHAR(15),
    severity           VARCHAR(10),
    risk_probability   FLOAT,
    uncertainty_range  FLOAT,
    description        TEXT,
    recommended_actions JSONB,
    acknowledged       BOOLEAN      DEFAULT FALSE,
    acknowledged_by    VARCHAR(100),
    acknowledged_at    TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- ML model registry (mirrors MLflow but provides quick DB lookups)
-- ---------------------------------------------------------------------------
CREATE TABLE model_registry (
    id             SERIAL PRIMARY KEY,
    model_name     VARCHAR(100)  NOT NULL,
    version        VARCHAR(50)   NOT NULL,
    mlflow_run_id  VARCHAR(50),
    region_code    VARCHAR(20),
    metrics        JSONB,
    is_active      BOOLEAN       DEFAULT FALSE,
    promoted_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);

COMMIT;
