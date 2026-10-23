import json                     
import numpy as np             
from pathlib import Path
from sklearn.ensemble import IsolationForest   
import joblib                   # to save the trained forest to a file

CALIB_DIR = Path(__file__).resolve().parent.parent / 'calibration_data'
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# the 5 numbers we use from every frame
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']

#########gathering the data from the calibration file for a user
def gather_data(user):

    path = CALIB_DIR / f'calibration_{user}.json' ####open the calibration file to get the data for the user
    with open(path, encoding='utf-8') as calibration_file:
        data = json.load(calibration_file)

    # 2. go through every run, and every frame in that run
    rows_sample = []
    for session in data['sessions']: ##session looping
        for frame in session['samples']: ####frame looping inside the session
            # 3. keep only the 5 numbers of this frame
            rows_sample.append([frame[feature_name] for feature_name in FEATURES])

    # 4. turn the list into a table: one row per frame, 5 columns
    feature_table  = np.array(rows_sample, dtype=float) ####OUTPUT
    return feature_table

#####################################################################################


######### training the isolation forest for a user
def train(user):
   
    feature_table = gather_data(user) # 1. get the table of calibration frames
    
    # print("Feature Table:", feature_table)
    print("Sample Length", len(feature_table))

     #setup the model
    forest_model = IsolationForest(
        n_estimators=200,      # build 200 trees
        contamination=0.03,    # the weirdest 3% of calibration frames count as unusual
        random_state=42)       # use the same random cuts every time we train

    #build the 200 trees from the calibration frames
    forest_model.fit(feature_table)

    # 4. check the result: ask the forest about its own calibration frames
    predictions = forest_model.predict(feature_table)    # one answer per frame: 1 = normal, -1 = unusual
    unusual_count = (predictions == -1).sum()            # count the -1 answers
    # print('frames:', len(feature_table), '| unusual:', unusual_count)

    # 5. save the trained forest to its own file
    MODELS_DIR.mkdir(exist_ok=True)                      # make the folder if it's missing
    model_path = MODELS_DIR / f'isolation_{user}.joblib'
    joblib.dump(forest_model, model_path)                # write the forest into the file
    print('saved to:', model_path)
    return forest_model


######### load the saved forest for a user 
def load(user):
    model_path = MODELS_DIR / f'isolation_{user}.joblib'
    if not model_path.exists():                     # this student has no trained forest yet
        print(f'No forest for "{user}" yet.')
        return None
    print(f'Loaded forest for "{user}".')
    return joblib.load(model_path)


######### check one frame: True = unusual, False = normal
def is_unusual(trained_forest, camera_frame):
    if trained_forest is None:                        # no forest -> never say unusual
        return False
    answer = trained_forest.predict([camera_frame])[0]   # 1 = normal, -1 = unusual
    return answer == -1

######### explain one frame: its score, its average path, and the cut-off path
def explain(trained_forest, camera_frame):
    # same math as isolation_report.py, but for ONE frame
    frames_per_tree = trained_forest.max_samples_
    normal_path = 2 * (np.log(frames_per_tree - 1) + 0.5772156649) - 2 * (frames_per_tree - 1) / frames_per_tree

    anomaly_score = -trained_forest.score_samples([camera_frame])[0]   # 0 to 1, higher = more unusual
    frame_path = -np.log2(anomaly_score) * normal_path                 # this frame's average path over the 200 trees

    cutoff_score = -trained_forest.offset_                             # the cut-off that contamination made
    cutoff_path = -np.log2(cutoff_score) * normal_path                 # the same cut-off as a path length

    return anomaly_score, frame_path, cutoff_path
# test: runs only when you start this file directly, not when the app imports it
if __name__ == '__main__':

    forest_model = load('demo')
    feature_table  = gather_data('demo')
    print('number of trees:', len(forest_model.estimators_))
    print('frames per tree:', forest_model.max_samples_)
    print('table size:', feature_table.shape)
    print('first frame:', feature_table[0])