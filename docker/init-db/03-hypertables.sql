BEGIN;

-- ---------------------------------------------------------------------------
-- Convert time-series tables to TimescaleDB hypertables
-- ---------------------------------------------------------------------------
SELECT create_hypertable('weather_events',      'event_time',   chunk_time_interval => INTERVAL '1 month', migrate_data => true);
SELECT create_hypertable('outage_observations', 'observed_at',  chunk_time_interval => INTERVAL '1 month', migrate_data => true);
SELECT create_hypertable('grid_load',           'recorded_at',  chunk_time_interval => INTERVAL '1 month', migrate_data => true);
SELECT create_hypertable('feature_store',       'computed_at',  chunk_time_interval => INTERVAL '1 month', migrate_data => true);
SELECT create_hypertable('predictions',         'predicted_at', chunk_time_interval => INTERVAL '1 month', migrate_data => true);

-- ---------------------------------------------------------------------------
-- Indexes: H3 cell lookups
-- ---------------------------------------------------------------------------
CREATE INDEX idx_weather_events_h3_res7      ON weather_events      (h3_index_res7, event_time DESC);
CREATE INDEX idx_weather_events_h3_res9      ON weather_events      (h3_index_res9, event_time DESC);
CREATE INDEX idx_outage_observations_h3      ON outage_observations (h3_index_res7, observed_at DESC);
CREATE INDEX idx_feature_store_h3            ON feature_store       (h3_index_res7, computed_at DESC);
CREATE INDEX idx_predictions_h3              ON predictions         (h3_index_res7, predicted_at DESC);
CREATE INDEX idx_infrastructure_h3           ON infrastructure      (h3_index_res7);
CREATE INDEX idx_socioeconomic_h3            ON socioeconomic       (h3_index_res7);

-- ---------------------------------------------------------------------------
-- Indexes: Event type and source filtering
-- ---------------------------------------------------------------------------
CREATE INDEX idx_weather_events_type         ON weather_events      (event_type, event_time DESC);
CREATE INDEX idx_weather_events_source       ON weather_events      (source, event_time DESC);
CREATE INDEX idx_weather_events_county       ON weather_events      (state_fips, county_fips, event_time DESC);
CREATE INDEX idx_outage_observations_county  ON outage_observations (state_fips, county_fips, observed_at DESC);
CREATE INDEX idx_grid_load_region            ON grid_load           (region_code, recorded_at DESC);
CREATE INDEX idx_predictions_region          ON predictions         (region_code, predicted_at DESC);

-- ---------------------------------------------------------------------------
-- Indexes: GIS spatial queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_weather_events_location_gist ON weather_events  USING GIST (location);
CREATE INDEX idx_regions_boundary_gist        ON regions         USING GIST (boundary);

-- ---------------------------------------------------------------------------
-- Indexes: Active / unacknowledged alerts (composite)
-- ---------------------------------------------------------------------------
CREATE INDEX idx_alerts_active ON alerts (severity, created_at DESC)
    WHERE acknowledged = FALSE AND (expires_at IS NULL OR expires_at > NOW());

CREATE INDEX idx_alerts_region ON alerts (region_code, created_at DESC);
CREATE INDEX idx_alerts_h3     ON alerts (h3_index_res7, created_at DESC);

-- ---------------------------------------------------------------------------
-- Indexes: Model registry active lookup
-- ---------------------------------------------------------------------------
CREATE INDEX idx_model_registry_active ON model_registry (region_code, model_name)
    WHERE is_active = TRUE;

-- ---------------------------------------------------------------------------
-- Indexes: Feature store version lookup
-- ---------------------------------------------------------------------------
CREATE INDEX idx_feature_store_version ON feature_store (feature_version, computed_at DESC);

COMMIT;
