import os
from pathlib import Path

PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')
OBSERVACIONS_DIR = Path(os.path.join(DATA_DIR, 'observacions'))
OBSERVACIONS_FILTRAT_DIR = Path(os.path.join(DATA_DIR, 'observacions_filtrat'))
ESTACIONS_DIR = Path(os.path.join(DATA_DIR, 'estacions_cabal.csv'))
TRAIN_TEST_INDEX_DIR = Path(os.path.join(DATA_DIR, 'splits_train_test'))
TRAIN_TEST_SERIES_DIR = Path(os.path.join(DATA_DIR, 'splits_train_test_series'))
OPTIMIZATION_DIR = Path(os.path.join(DATA_DIR, 'optimization'))