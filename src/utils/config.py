import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
METRICS_DIR    = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR      = os.path.join(RESULTS_DIR, "plots")
CONFIGS_DIR    = os.path.join(BASE_DIR, "configs")

SEED = 42

SEQ_LEN       = 365
FORECAST_STEP = 1

TRAIN_START = "1980-10-01"
TRAIN_END   = "2000-09-30"
VAL_START   = "2000-10-01"
VAL_END     = "2005-09-30"
TEST_START  = "2005-10-01"
TEST_END    = "2014-09-30"

LAG_DAYS = [1, 2, 3, 7, 14, 30]

DYNAMIC_FEATURES = [
    "total_precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum",
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
]

TARGET_COL = "streamflow"

STATIC_ATTRIBUTES = [
    "area",          # catchment area (km2), from attributes_other
    "ele_mt_sav",    # mean elevation, from hydroatlas
    "slp_dg_sav",    # mean slope (degrees), from hydroatlas
    "for_pc_sse",    # forest fraction, from hydroatlas
    "frac_snow",     # snow fraction, from attributes_caravan
    "p_mean",        # mean precipitation, from attributes_caravan
    "pet_mean",      # mean PET, from attributes_caravan
    "aridity",       # aridity index, from attributes_caravan
    "high_prec_freq",# high precip frequency, from attributes_caravan
]

MAX_NAN_FRACTION = 0.05
MAX_GAP_FILL_DAYS = 3

GLOBAL_MODEL = True

DL_DEFAULTS = dict(
    hidden_size=256,
    num_layers=2,
    dropout=0.4,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=50,
    patience=10,
    grad_clip=1.0,
    static_embed_dim=32,
)
