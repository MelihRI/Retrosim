"""
Regulatory Monitoring Agent
=============================

Tracks and projects maritime regulatory changes relevant to investment decisions:

1. IMO CII (Carbon Intensity Indicator) thresholds — yearly tightening schedule
2. EU ETS (Emissions Trading System) carbon prices — 2025-2050 projection
3. EEDI Phase requirements
4. Upcoming regulation change detection & alerts

Data Sources:
  - Embedded baseline tables (works offline)
  - Optional API fetch for live EU ETS prices (ECX/ICE)
  - IMO MEPC resolution timelines

Usage:
    agent = RegulatoryAgent()
    cii = agent.get_cii_threshold(year=2030, ship_type='bulk_carrier', dwt=55000)
    ets = agent.get_carbon_price(year=2035, scenario='accelerated')
    projection = agent.project_regulatory_costs(vessel_data, horizon=20)
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from PyQt6.QtCore import QObject, pyqtSignal


# ═══════════════════════════════════════════════════════════════════════════════
# Embedded Regulatory Data Tables
# ═══════════════════════════════════════════════════════════════════════════════

# IMO CII Reference Lines (MEPC.354(78)) — gCO2/dwt·nm
# Rating boundaries: A, B, C, D, E
# These are reduction factors from 2019 baseline
CII_REDUCTION_FACTORS = {
    2023: 0.05,   # 5% reduction from 2019
    2024: 0.07,
    2025: 0.09,
    2026: 0.11,
    2027: 0.13,   # Projected
    2028: 0.15,
    2029: 0.17,
    2030: 0.20,   # Significant tightening expected
    2031: 0.22,
    2032: 0.24,
    2033: 0.26,
    2034: 0.28,
    2035: 0.30,
    2040: 0.40,   # Interpolated
    2045: 0.50,
    2050: 0.60,
}

# CII Reference values by ship type (gCO2 / dwt·nm) — 2019 baseline
CII_REFERENCE_2019 = {
    'bulk_carrier':     {  # DWT ranges
        (0, 10000):     17.5,
        (10000, 25000): 12.5,
        (25000, 50000): 10.0,
        (50000, 100000): 8.5,
        (100000, 999999): 7.5,
    },
    'tanker': {
        (0, 10000):     18.0,
        (10000, 25000): 13.0,
        (25000, 50000): 10.5,
        (50000, 100000): 9.0,
        (100000, 999999): 8.0,
    },
    'container': {
        (0, 10000):     20.0,
        (10000, 25000): 15.0,
        (25000, 50000): 12.0,
        (50000, 100000): 10.0,
        (100000, 999999): 8.5,
    },
    'general_cargo': {
        (0, 10000):     15.0,
        (10000, 25000): 11.0,
        (25000, 50000):  9.0,
        (50000, 999999): 7.5,
    },
    'koster': {  # Turkish coaster (our primary target)
        (0, 5000):      16.0,
        (5000, 10000):  13.0,
        (10000, 999999): 11.0,
    },
}

# CII Rating bands (d-values as factors around reference line)
CII_RATING_BANDS = {
    'A': (-np.inf, 0.83),    # Superior
    'B': (0.83, 0.94),       # Minor superior
    'C': (0.94, 1.06),       # Moderate (target minimum)
    'D': (1.06, 1.19),       # Inferior
    'E': (1.19, np.inf),     # Much inferior
}

# EU ETS Carbon Prices (€/tCO2) — 3 scenarios
EU_ETS_PRICES = {
    'baseline': {
        2024:  65, 2025:  70, 2026: 75, 2027: 80, 2028: 85,
        2029:  90, 2030: 100, 2032: 110, 2035: 130,
        2040: 160, 2045: 190, 2050: 220,
    },
    'accelerated': {
        2024:  65, 2025:  80, 2026: 95, 2027: 110, 2028: 125,
        2029: 140, 2030: 160, 2032: 190, 2035: 250,
        2040: 350, 2045: 450, 2050: 500,
    },
    'paris_aligned': {
        2024:  65, 2025:  90, 2026: 115, 2027: 140, 2028: 170,
        2029: 200, 2030: 250, 2032: 320, 2035: 400,
        2040: 550, 2045: 700, 2050: 800,
    },
}

# Maritime ETS phase-in (% of emissions covered)
EU_ETS_MARITIME_PHASEIN = {
    2024: 0.40,   # 40% of emissions
    2025: 0.70,   # 70%
    2026: 1.00,   # 100%
}

# EEDI Phase requirements
EEDI_PHASES = {
    'phase_0': {'start': 2013, 'end': 2014, 'reduction': 0.00},
    'phase_1': {'start': 2015, 'end': 2019, 'reduction': 0.10},
    'phase_2': {'start': 2020, 'end': 2024, 'reduction': 0.20},
    'phase_3': {'start': 2025, 'end': 2030, 'reduction': 0.30},
    'phase_4': {'start': 2030, 'end': 2050, 'reduction': 0.50},  # Projected
}

# CO2 emission factors by fuel type (tCO2 / tFuel)
EMISSION_FACTORS = {
    'HFO':  3.114,
    'MDO':  3.206,
    'MGO':  3.206,
    'LNG':  2.750,
    'LPG':  3.000,
    'methanol': 1.375,
    'ammonia':  0.000,
    'hydrogen': 0.000,
}


# ═══════════════════════════════════════════════════════════════════════════════
# RegulatoryAgent
# ═══════════════════════════════════════════════════════════════════════════════

class RegulatoryAgent(QObject):
    """
    Maritime Regulatory Monitoring & Projection Agent.

    Provides:
      - CII classification for any year/vessel combination
      - EU ETS carbon cost projections (3 scenarios)
      - EEDI compliance checking
      - Regulatory change alerts
    """

    progress_signal = pyqtSignal(int, str)
    alert_signal    = pyqtSignal(str, str)   # (severity, message)
    data_updated    = pyqtSignal(dict)       # live price updates

    CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'regulatory_cache')

    def __init__(self):
        super().__init__()
        self._ets_cache: Dict = {}
        self._last_fetch: Optional[datetime] = None
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._load_cache()

    # ──────────────────────────────────────────────────────────────────────
    # CII Classification
    # ──────────────────────────────────────────────────────────────────────

    def get_cii_threshold(self, year: int, ship_type: str = 'koster',
                          dwt: float = 5000) -> Dict:
        """
        Calculate CII thresholds for a given year and vessel.

        Args:
            year: Target year (2023-2050)
            ship_type: Vessel type key
            dwt: Deadweight tonnage

        Returns:
            {
                'reference': float,    # gCO2/(dwt·nm) baseline
                'required': float,     # with reduction applied
                'reduction_pct': float,
                'ratings': dict,       # {A: max, B: max, C: max, D: max}
            }
        """
        # Get 2019 baseline
        type_refs = CII_REFERENCE_2019.get(ship_type, CII_REFERENCE_2019.get('koster'))
        ref_val = 13.0  # fallback
        for (dwt_min, dwt_max), ref in type_refs.items():
            if dwt_min <= dwt < dwt_max:
                ref_val = ref
                break

        # Get reduction factor (interpolate between known years)
        reduction = self._interpolate_factor(year, CII_REDUCTION_FACTORS)

        required = ref_val * (1.0 - reduction)

        # Rating boundaries
        ratings = {}
        for grade, (lo, hi) in CII_RATING_BANDS.items():
            ratings[grade] = (required * lo, required * hi)

        return {
            'year': year,
            'ship_type': ship_type,
            'dwt': dwt,
            'reference_2019': ref_val,
            'reduction_pct': reduction * 100,
            'required_cii': required,
            'ratings': ratings,
        }

    def classify_cii(self, attained_cii: float, year: int,
                     ship_type: str = 'koster', dwt: float = 5000) -> str:
        """
        Classify vessel's CII rating.

        Args:
            attained_cii: Actual CII in gCO2/(dwt·nm)
            year: Assessment year
            ship_type: Vessel type
            dwt: Deadweight tonnage

        Returns:
            Rating grade: 'A', 'B', 'C', 'D', or 'E'
        """
        thresholds = self.get_cii_threshold(year, ship_type, dwt)
        required = thresholds['required_cii']
        ratio = attained_cii / required if required > 0 else 999

        for grade, (lo, hi) in CII_RATING_BANDS.items():
            if lo <= ratio < hi:
                return grade
        return 'E'

    # ──────────────────────────────────────────────────────────────────────
    # EU ETS Carbon Pricing
    # ──────────────────────────────────────────────────────────────────────

    def get_carbon_price(self, year: int, scenario: str = 'baseline') -> float:
        """
        Get projected EU ETS carbon price for a given year.

        Args:
            year: Target year (2024-2050)
            scenario: 'baseline', 'accelerated', or 'paris_aligned'

        Returns:
            Price in €/tCO2
        """
        prices = EU_ETS_PRICES.get(scenario, EU_ETS_PRICES['baseline'])
        return self._interpolate_factor(year, prices)

    def get_maritime_ets_coverage(self, year: int) -> float:
        """Get percentage of maritime emissions covered by EU ETS."""
        if year < 2024:
            return 0.0
        if year >= 2026:
            return 1.0
        return EU_ETS_MARITIME_PHASEIN.get(year, 1.0)

    def calculate_annual_ets_cost(self, fuel_consumption_tonnes: float,
                                   fuel_type: str, year: int,
                                   scenario: str = 'baseline') -> Dict:
        """
        Calculate annual EU ETS cost for a vessel.

        Args:
            fuel_consumption_tonnes: Annual fuel consumption in tonnes
            fuel_type: 'HFO', 'MDO', 'LNG', etc.
            year: Assessment year
            scenario: ETS price scenario

        Returns:
            {
                'co2_emissions': float (tCO2),
                'carbon_price': float (€/tCO2),
                'coverage': float (0-1),
                'ets_cost': float (€),
            }
        """
        ef = EMISSION_FACTORS.get(fuel_type, EMISSION_FACTORS['HFO'])
        co2 = fuel_consumption_tonnes * ef
        price = self.get_carbon_price(year, scenario)
        coverage = self.get_maritime_ets_coverage(year)

        return {
            'co2_emissions_t': co2,
            'carbon_price_eur': price,
            'coverage_pct': coverage * 100,
            'ets_cost_eur': co2 * price * coverage,
            'fuel_type': fuel_type,
            'year': year,
            'scenario': scenario,
        }

    # ──────────────────────────────────────────────────────────────────────
    # EEDI Compliance
    # ──────────────────────────────────────────────────────────────────────

    def get_eedi_requirement(self, year: int) -> Dict:
        """Get current EEDI phase and reduction requirement."""
        for phase_name, info in EEDI_PHASES.items():
            if info['start'] <= year <= info['end']:
                return {
                    'phase': phase_name,
                    'reduction_pct': info['reduction'] * 100,
                    'year': year,
                }
        # Beyond 2050
        return {'phase': 'phase_4+', 'reduction_pct': 50.0, 'year': year}

    # ──────────────────────────────────────────────────────────────────────
    # Regulation Change Detection
    # ──────────────────────────────────────────────────────────────────────

    def detect_regulation_changes(self, current_year: int,
                                   horizon_years: int = 5) -> List[Dict]:
        """
        Detect significant regulation changes in the coming years.

        Returns a list of alerts for changes that exceed 5% impact.
        """
        alerts = []

        for future_year in range(current_year + 1, current_year + horizon_years + 1):
            # CII tightening
            curr_red = self._interpolate_factor(current_year, CII_REDUCTION_FACTORS) * 100
            fut_red = self._interpolate_factor(future_year, CII_REDUCTION_FACTORS) * 100

            if (fut_red - curr_red) > 5:
                alerts.append({
                    'type': 'CII_TIGHTENING',
                    'year': future_year,
                    'severity': 'WARNING',
                    'message': f"CII reduction jumps from {curr_red:.0f}% to {fut_red:.0f}% "
                               f"in {future_year}. Vessels may drop rating.",
                    'impact_pct': fut_red - curr_red,
                })

            # ETS price jump
            for scenario in ['baseline', 'accelerated']:
                curr_price = self.get_carbon_price(current_year, scenario)
                fut_price = self.get_carbon_price(future_year, scenario)
                pct_change = ((fut_price - curr_price) / curr_price * 100
                              if curr_price > 0 else 0)

                if pct_change > 20:
                    alerts.append({
                        'type': 'ETS_PRICE_JUMP',
                        'year': future_year,
                        'scenario': scenario,
                        'severity': 'CAUTION' if pct_change > 40 else 'WARNING',
                        'message': f"EU ETS price ({scenario}): €{curr_price:.0f} → "
                                   f"€{fut_price:.0f} (+{pct_change:.0f}%) by {future_year}.",
                        'impact_pct': pct_change,
                    })

            # ETS maritime phase-in
            curr_cov = self.get_maritime_ets_coverage(current_year)
            fut_cov = self.get_maritime_ets_coverage(future_year)
            if fut_cov > curr_cov:
                alerts.append({
                    'type': 'ETS_PHASEIN',
                    'year': future_year,
                    'severity': 'IMPORTANT',
                    'message': f"EU ETS maritime coverage: {curr_cov*100:.0f}% → "
                               f"{fut_cov*100:.0f}% in {future_year}.",
                    'impact_pct': (fut_cov - curr_cov) * 100,
                })

        return alerts

    # ──────────────────────────────────────────────────────────────────────
    # 20-Year Projection
    # ──────────────────────────────────────────────────────────────────────

    def project_regulatory_costs(self, vessel_data: Dict,
                                  horizon: int = 20,
                                  scenario: str = 'baseline') -> Dict:
        """
        Project regulatory costs over a 20-year horizon.

        Args:
            vessel_data: Vessel parameters dict
            horizon: Projection years
            scenario: ETS price scenario

        Returns:
            {
                'years': [2025, ...],
                'ets_costs': [€, ...],
                'cii_ratings': ['C', ...],
                'total_regulatory_cost': float,
                'alerts': [...]
            }
        """
        current_year = datetime.now().year
        dwt = float(vessel_data.get('dwt', 5000))
        fuel_type = str(vessel_data.get('fuel_type', 'HFO'))
        if fuel_type.isdigit():
            fuel_type = {0: 'HFO', 1: 'MDO', 2: 'LNG'}.get(int(fuel_type), 'HFO')

        # Estimate annual fuel consumption
        engine_power = float(vessel_data.get('engine_power', 2000))
        sfoc = float(vessel_data.get('sfoc', 180))  # g/kWh
        operating_hours = 6000  # typical hours/year
        annual_fuel_t = engine_power * sfoc * operating_hours / 1e6

        years = list(range(current_year, current_year + horizon + 1))
        ets_costs = []
        cii_ratings = []
        cii_values = []

        # Baseline attained CII (estimate)
        speed = float(vessel_data.get('speed', 12))
        co2_factor = EMISSION_FACTORS.get(fuel_type, 3.114)
        transport_work = dwt * speed * 0.5144 * operating_hours  # t·nm approximation
        base_cii = (annual_fuel_t * co2_factor * 1e6) / transport_work if transport_work > 0 else 15.0

        age = float(vessel_data.get('age', 10))

        for i, year in enumerate(years):
            # Aging penalty: +1.5% CII degradation per year
            vessel_age = age + i
            aging_factor = 1.0 + 0.015 * vessel_age
            attained_cii = base_cii * aging_factor

            # CII classification
            rating = self.classify_cii(attained_cii, year, 'koster', dwt)
            cii_ratings.append(rating)
            cii_values.append(attained_cii)

            # ETS cost
            ets = self.calculate_annual_ets_cost(annual_fuel_t, fuel_type, year, scenario)
            ets_costs.append(ets['ets_cost_eur'])

        return {
            'years': years,
            'ets_costs_eur': ets_costs,
            'cii_ratings': cii_ratings,
            'cii_values': cii_values,
            'total_ets_cost': sum(ets_costs),
            'annual_fuel_tonnes': annual_fuel_t,
            'scenario': scenario,
            'alerts': self.detect_regulation_changes(current_year, horizon),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Live ETS Price Fetch (Optional)
    # ──────────────────────────────────────────────────────────────────────

    def fetch_live_ets_price(self) -> Optional[float]:
        """
        Attempt to fetch current EU ETS carbon price from public API.

        Returns €/tCO2 or None if fetch fails.
        """
        try:
            import requests
            # Using a public proxy endpoint for EU ETS futures
            resp = requests.get(
                "https://api.ember-climate.org/v1/carbon-price"
                "?entity=European%20Union&is_current=true",
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and len(data['data']) > 0:
                    price = float(data['data'][0].get('price', 0))
                    self._ets_cache['live_price'] = price
                    self._ets_cache['fetch_time'] = datetime.now().isoformat()
                    self._save_cache()
                    self.data_updated.emit({'live_ets_price': price})
                    return price
        except Exception as e:
            print(f"ℹ️ Live ETS fetch skipped: {e}")

        # Fallback to cache
        return self._ets_cache.get('live_price')

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpolate_factor(year: int, lookup: Dict[int, float]) -> float:
        """Linear interpolation between lookup table years."""
        years = sorted(lookup.keys())
        if year <= years[0]:
            return lookup[years[0]]
        if year >= years[-1]:
            return lookup[years[-1]]

        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                t = (year - years[i]) / (years[i + 1] - years[i])
                return lookup[years[i]] * (1 - t) + lookup[years[i + 1]] * t

        return lookup[years[-1]]

    def _load_cache(self):
        """Load cached data from disk."""
        cache_path = os.path.join(self.CACHE_DIR, 'ets_cache.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    self._ets_cache = json.load(f)
            except Exception:
                pass

    def _save_cache(self):
        """Save cache to disk."""
        try:
            cache_path = os.path.join(self.CACHE_DIR, 'ets_cache.json')
            with open(cache_path, 'w') as f:
                json.dump(self._ets_cache, f, indent=2)
        except Exception:
            pass

    def get_summary(self, vessel_data: Dict, year: int = None) -> Dict:
        """
        Get a quick regulatory summary for the status bar.

        Returns:
            {
                'cii_rating': 'C',
                'ets_price': 100.0,
                'eedi_phase': 'phase_3',
                'alerts_count': 2,
            }
        """
        if year is None:
            year = datetime.now().year

        dwt = float(vessel_data.get('dwt', 5000))

        # Quick CII
        cii_info = self.get_cii_threshold(year, 'koster', dwt)

        # ETS
        ets_price = self.get_carbon_price(year)
        live_price = self._ets_cache.get('live_price')
        if live_price:
            ets_price = live_price

        # EEDI
        eedi_info = self.get_eedi_requirement(year)

        # Alerts
        alerts = self.detect_regulation_changes(year, 3)

        return {
            'year': year,
            'cii_required': cii_info['required_cii'],
            'cii_reduction_pct': cii_info['reduction_pct'],
            'ets_price_eur': ets_price,
            'ets_live': live_price is not None,
            'eedi_phase': eedi_info['phase'],
            'eedi_reduction_pct': eedi_info['reduction_pct'],
            'alerts_count': len(alerts),
            'alerts': alerts[:3],  # top 3
        }
