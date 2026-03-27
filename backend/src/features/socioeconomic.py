"""Socioeconomic and Vulnerability Feature Engineering.

Computes community vulnerability indices from Census data,
identifying areas where outages would have disproportionate impact.
"""

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class SocioeconomicFeatureBuilder:
    """Builds vulnerability features from Census-derived data."""

    def vulnerability_index(self, socio_data: dict | None) -> dict[str, float]:
        """Compute composite vulnerability index from socioeconomic indicators.

        The vulnerability index captures areas where outages cause
        greater harm: dense populations, lower-income communities,
        older housing stock, and proximity to critical facilities.

        Args:
            socio_data: Dict with population_density, median_income,
                       housing_age_median, critical_facilities_count.

        Returns:
            Dict of socioeconomic features including composite index.
        """
        if socio_data is None:
            return {
                "population_density": 0.0,
                "median_income_normalized": 0.0,
                "housing_vulnerability": 0.0,
                "critical_facility_density": 0.0,
                "composite_vulnerability_index": 0.0,
            }

        pop_density = float(socio_data.get("population_density", 0) or 0)
        median_income = float(socio_data.get("median_income", 0) or 0)
        housing_age = float(socio_data.get("housing_age_median", 0) or 0)
        critical_facilities = float(socio_data.get("critical_facilities_count", 0) or 0)

        # Normalize to [0, 1] using reasonable US ranges
        pop_score = min(1.0, pop_density / 5000.0)  # 5000 ppl/km^2 ~ dense urban
        income_score = 1.0 - min(1.0, median_income / 100_000.0)  # lower income = higher vuln
        housing_score = min(1.0, housing_age / 60.0)  # older housing = higher vuln
        facility_score = min(1.0, critical_facilities / 10.0)  # more facilities = higher impact

        # Weighted composite: population and critical facilities matter most
        weights = {"pop": 0.30, "income": 0.20, "housing": 0.15, "facility": 0.35}
        composite = (
            weights["pop"] * pop_score
            + weights["income"] * income_score
            + weights["housing"] * housing_score
            + weights["facility"] * facility_score
        )

        return {
            "population_density": pop_density,
            "median_income_normalized": median_income / 100_000.0 if median_income > 0 else 0.0,
            "housing_vulnerability": housing_score,
            "critical_facility_density": critical_facilities,
            "composite_vulnerability_index": float(np.clip(composite, 0.0, 1.0)),
        }
