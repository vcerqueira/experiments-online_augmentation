from utils.load_data.m4 import M4Dataset
from utils.load_data.m3 import M3Dataset
from utils.load_data.tourism import TourismDataset
from utils.load_data.misc import MiscDataset
from utils.load_data.gluonts import GluontsDataset

DATASETS = {
    'M4': M4Dataset,
    'M3': M3Dataset,
    'Tourism': TourismDataset,
    'Misc': MiscDataset,
    'Gluonts': GluontsDataset,
}

DATA_GROUPS = {
    'M3': ['Monthly', 'Quarterly'],
    'M4': ['Monthly', 'Quarterly'],
    'Tourism': ['Monthly', 'Quarterly'],
    'Misc': ['NN3', 'AusDemandWeekly'],
    'Gluonts': ['m1_monthly', 'm1_quarterly', 'nn5_weekly'],
}

DATA_GROUPS_ = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
]
