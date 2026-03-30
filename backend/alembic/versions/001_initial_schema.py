"""Initial schema — matches docker/init-db SQL definitions.

Revision ID: 001_initial
Revises:
Create Date: 2026-03-30
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- Regions ---------------------------------------------------------------
    op.create_table(
        "regions",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("code", sa.String(20), nullable=False, unique=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("config_path", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Weather Events --------------------------------------------------------
    op.create_table(
        "weather_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("event_type", sa.String(60), nullable=True),
        sa.Column("magnitude", sa.Float(), nullable=True),
        sa.Column("magnitude_type", sa.String(10), nullable=True),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("h3_index_res9", sa.String(15), nullable=True),
        sa.Column("state_fips", sa.String(5), nullable=True),
        sa.Column("county_fips", sa.String(5), nullable=True),
        sa.Column("damage_property", sa.Numeric(), nullable=True),
        sa.Column("damage_crops", sa.Numeric(), nullable=True),
        sa.Column("injuries", sa.Integer(), server_default="0"),
        sa.Column("deaths", sa.Integer(), server_default="0"),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("episode_id", sa.BigInteger(), nullable=True),
        sa.Column("raw_data", JSONB(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Outage Observations ---------------------------------------------------
    op.create_table(
        "outage_observations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("county_fips", sa.String(5), nullable=True),
        sa.Column("state_fips", sa.String(2), nullable=True),
        sa.Column("customers_out", sa.Integer(), nullable=True),
        sa.Column("total_customers", sa.Integer(), nullable=True),
        sa.Column("outage_fraction", sa.Float(), nullable=True),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("source", sa.String(20), server_default="eagle_i"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Grid Load -------------------------------------------------------------
    op.create_table(
        "grid_load",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("region_code", sa.String(20), nullable=True),
        sa.Column("load_mw", sa.Float(), nullable=True),
        sa.Column("capacity_mw", sa.Float(), nullable=True),
        sa.Column("reserve_margin_pct", sa.Float(), nullable=True),
        sa.Column("frequency_hz", sa.Float(), nullable=True),
        sa.Column("source", sa.String(20), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Infrastructure --------------------------------------------------------
    op.create_table(
        "infrastructure",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("county_fips", sa.String(5), nullable=True),
        sa.Column("transmission_line_km", sa.Float(), server_default="0"),
        sa.Column("distribution_line_km", sa.Float(), server_default="0"),
        sa.Column("substations_count", sa.Integer(), server_default="0"),
        sa.Column("avg_line_age_years", sa.Float(), nullable=True),
        sa.Column("vegetation_density", sa.Float(), nullable=True),
        sa.Column("land_cover_type", sa.String(30), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Socioeconomic ---------------------------------------------------------
    op.create_table(
        "socioeconomic",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("county_fips", sa.String(5), nullable=True),
        sa.Column("population_density", sa.Float(), nullable=True),
        sa.Column("median_income", sa.Float(), nullable=True),
        sa.Column("critical_facilities_count", sa.Integer(), server_default="0"),
        sa.Column("housing_age_median", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Feature Store ---------------------------------------------------------
    op.create_table(
        "feature_store",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("feature_version", sa.String(10), nullable=True),
        sa.Column("features", JSONB(), nullable=True),
        sa.Column("target_outage_occurred", sa.Boolean(), nullable=True),
        sa.Column("target_customers_affected", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Predictions -----------------------------------------------------------
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("region_code", sa.String(20), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("risk_probability", sa.Float(), nullable=True),
        sa.Column("uncertainty_lower", sa.Float(), nullable=True),
        sa.Column("uncertainty_upper", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(10), nullable=True),
        sa.Column("features_snapshot", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )

    # -- Alerts ----------------------------------------------------------------
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("region_code", sa.String(20), nullable=True),
        sa.Column("h3_index_res7", sa.String(15), nullable=True),
        sa.Column("severity", sa.String(10), nullable=True),
        sa.Column("risk_probability", sa.Float(), nullable=True),
        sa.Column("uncertainty_range", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("recommended_actions", JSONB(), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), server_default="false"),
        sa.Column("acknowledged_by", sa.String(100), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -- Model Registry --------------------------------------------------------
    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("mlflow_run_id", sa.String(50), nullable=True),
        sa.Column("region_code", sa.String(20), nullable=True),
        sa.Column("metrics", JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("model_name", "version", name="uq_model_name_version"),
    )

    # -- Indexes (matching 03-hypertables.sql minus TimescaleDB-specific) ------
    op.create_index("idx_weather_events_h3_res7", "weather_events", ["h3_index_res7", sa.text("event_time DESC")])
    op.create_index("idx_outage_observations_h3", "outage_observations", ["h3_index_res7", sa.text("observed_at DESC")])
    op.create_index("idx_feature_store_h3", "feature_store", ["h3_index_res7", sa.text("computed_at DESC")])
    op.create_index("idx_predictions_h3", "predictions", ["h3_index_res7", sa.text("predicted_at DESC")])
    op.create_index("idx_infrastructure_h3", "infrastructure", ["h3_index_res7"])
    op.create_index("idx_socioeconomic_h3", "socioeconomic", ["h3_index_res7"])
    op.create_index("idx_grid_load_region", "grid_load", ["region_code", sa.text("recorded_at DESC")])
    op.create_index("idx_predictions_region", "predictions", ["region_code", sa.text("predicted_at DESC")])
    op.create_index("idx_alerts_region", "alerts", ["region_code", sa.text("created_at DESC")])


def downgrade() -> None:
    op.drop_table("model_registry")
    op.drop_table("alerts")
    op.drop_table("predictions")
    op.drop_table("feature_store")
    op.drop_table("socioeconomic")
    op.drop_table("infrastructure")
    op.drop_table("grid_load")
    op.drop_table("outage_observations")
    op.drop_table("weather_events")
    op.drop_table("regions")
