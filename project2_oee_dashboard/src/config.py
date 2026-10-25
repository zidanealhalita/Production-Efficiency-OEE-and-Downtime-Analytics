"""Central configuration: palette, labels and default (editable) assumptions.

Everything an analyst might want to tweak is in ``DEFAULT_ASSUMPTIONS``. The Streamlit
sidebar exposes these values and passes the edited copy to every calculation, so a change
flows through all pages.
"""
from __future__ import annotations

DATA_DIR = "data_production"

# --- Colour-blind-safe palette (Okabe-Ito) -------------------------------------------------
LINE_COLORS = {
    "L01": "#0072B2",  # blue
    "L02": "#D55E00",  # vermillion
    "L03": "#009E73",  # bluish green
    "L04": "#CC79A7",  # reddish purple
    "L05": "#E69F00",  # orange
    "L06": "#56B4E9",  # sky blue
}
SHIFT_COLORS = {"S1": "#0072B2", "S2": "#56B4E9", "S3": "#D55E00"}
NEUTRAL = "#6B7280"
HIGHLIGHT = "#D55E00"
GOOD = "#009E73"
BAD = "#D55E00"

# --- OEE loss categories -------------------------------------------------------------------
LOSS_CATEGORIES = {  # slug -> label used in the data
    "breakdown": "Breakdown",
    "changeover": "Changeover",
    "material_shortage": "Material shortage",
    "quality_hold": "Quality hold",
    "utilities": "Utilities",
    "waiting_operator": "Waiting operator",
    "unplanned_cleaning": "Unplanned cleaning",
    "minor_stoppage": "Minor stoppage",
}
AVAILABILITY_CATEGORIES = [k for k in LOSS_CATEGORIES if k != "minor_stoppage"]
SLUG_BY_LABEL = {v: k for k, v in LOSS_CATEGORIES.items()}

LOSS_COLORS = {
    "Breakdown": "#D55E00",
    "Changeover": "#E69F00",
    "Material shortage": "#009E73",
    "Quality hold": "#CC79A7",
    "Utilities": "#56B4E9",
    "Waiting operator": "#0072B2",
    "Unplanned cleaning": "#8C6D31",
    "Minor stoppage": "#999999",
}

# --- Benchmark bands (common rule of thumb, NOT a Danone target) ---------------------------
OEE_WORLD_CLASS = 0.85
OEE_TYPICAL_LOW = 0.60

RAINY_MONTHS = (11, 12, 1, 2, 3)
L02_FIX_DATE = "2025-10-01"
L02_PROBLEM = ("2025-04-01", "2025-09-30")
L02_AFTER = ("2025-10-01", "2025-12-31")

# --- Default assumptions (all editable in the sidebar) -------------------------------------
# Money values are ILLUSTRATIVE planning assumptions, not company data.
SKUS = [
    "WTR-220", "WTR-600", "WTR-1500", "WTR-GAL19",
    "MLK-UHT200", "MLK-UHT1000", "FRM-400", "FRM-900",
]
DEFAULT_MARGIN_PER_UNIT = {  # IDR contribution margin per good unit
    "WTR-220": 250, "WTR-600": 450, "WTR-1500": 900, "WTR-GAL19": 4000,
    "MLK-UHT200": 1200, "MLK-UHT1000": 4500, "FRM-400": 15000, "FRM-900": 30000,
}
DEFAULT_UNIT_COST = {  # IDR variable cost lost per rejected unit (material + conversion)
    "WTR-220": 700, "WTR-600": 1300, "WTR-1500": 2800, "WTR-GAL19": 13000,
    "MLK-UHT200": 3000, "MLK-UHT1000": 11000, "FRM-400": 45000, "FRM-900": 95000,
}

# Improvement actions: id -> (title, default effort 1 (easy) .. 5 (hard))
ACTIONS = {
    "l02_filler": ("L02 Filler: sustain the overhaul + PM routine", 2),
    "smed_l05": ("SMED on L05 UHT changeovers", 3),
    "night_shift": ("Night-shift staffing & skills plan", 3),
    "l06_material": ("Supplier review for L06 material shortages", 2),
    "startup_idle": ("Start-up-after-idle checklist & warm-up run", 1),
    "rainy_utilities": ("Rainy-season utilities protection", 3),
    "minor_stoppage": ("Minor-stoppage elimination on top reasons", 2),
}

DEFAULT_ASSUMPTIONS = {
    "margin_per_unit": DEFAULT_MARGIN_PER_UNIT,
    "unit_cost": DEFAULT_UNIT_COST,
    "labour_rate_idr_per_h": 75_000,     # maintenance technician, fully loaded
    "grid_factor_kg_per_kwh": 0.85,       # assumed grid emission factor (same as dataset)
    "idle_energy_share": 0.20,            # energy of a stopped-but-powered line vs running
    "idle_water_share": 0.10,             # water use of a stopped line vs running
    "night_gap_closure": 0.50,            # share of night-shift OEE gap that can be closed
    "smed_reduction": 0.30,               # changeover time reduction
    "l06_avoidable_share": 0.70,          # share of excess shortage minutes a supplier fix removes
    "startup_fixable_share": 0.40,        # share of start-up excess losses a checklist removes
    "rainy_fixable_share": 0.50,          # share of rainy-season excess utility stops removed
    "minor_stop_reduction": 0.15,         # reduction of minor-stoppage minutes
    "effort": {k: v[1] for k, v in ACTIONS.items()},
}
