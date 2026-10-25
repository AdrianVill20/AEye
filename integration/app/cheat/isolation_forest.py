# Isolation Forest trained in Colab on NORMAL exam videos (sessions 01, 02, 05).
# One model for every student - it does not use calibration.
import joblib

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
    # feats = the app's feats dict for one frame; take the 5 numbers in the trained order
    camera_frame = [[feats[name] for name in FEATURES]]
    return forest.predict(camera_frame)[0] == -1     # -1 = unusual, 1 = normal