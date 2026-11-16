# Isolation Forest trained in Colab on NORMAL exam videos (sessions 01, 02, 05).
# One model for every student - it does not use calibration.
import joblib
import warnings

# the order below comes from the model itself, so a list without column names is safe
warnings.filterwarnings('ignore', message='X does not have valid feature names')

from paths import MODELS_DIR

MODEL_PATH = MODELS_DIR / 'baseline_01_02_05_isolation_forest.joblib'
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']   # same order as in Colab


def load(path=None):
    # path = which forest file to use; None = the baseline (live tracking)
    path = path or MODEL_PATH
    if not path.exists():
        print(f'[ISOLATION] No model at {path}')
        return None
    print(f'[ISOLATION] Using {path.name}')    # so the console shows which forest is running
    return joblib.load(path)

def is_unusual(forest, feats):
    # take the features this model was trained on, in its own order (4 for no-roll, 5 for the others)
    camera_frame = [[feats[name] for name in forest.feature_names_in_]]
    return forest.predict(camera_frame)[0] == -1     # -1 = unusual, 1 = normal