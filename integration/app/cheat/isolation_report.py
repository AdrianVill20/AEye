
import joblib              
import numpy as np              

from isolation_forest import gather_data, MODELS_DIR   

USER = 'demo'                

# 1. load the trained forest that train() saved in Stage 3
model_path = MODELS_DIR / f'isolation_{USER}.joblib'
forest_model = joblib.load(model_path)

# 2. get the same table of calibration frames (Stage 1)
feature_table = gather_data(USER)

print('loaded:', model_path)
print('frames:', len(feature_table))
print('trees:', len(forest_model.estimators_))

# 3. score every frame (walks each frame through all 200 trees)
raw_scores = forest_model.score_samples(feature_table) ##THIS IS WHAT THE MODEL GIVE US
anomaly_scores = -raw_scores                  # 0 to 1, higher = more unusual

# 4. turn every score into the frame's average path length (over the 200 trees)
frames_per_tree = forest_model.max_samples_
normal_path = 2 * (np.log(frames_per_tree - 1) + 0.5772156649) - 2 * (frames_per_tree - 1) / frames_per_tree ##from the isolation forest paper(Inventor's constant = 0.5772156649)


# 5. turn every score back into its average path length
average_paths = -np.log2(anomaly_scores) * normal_path

# 6. the cut-off line that contamination made during training
raw_cutoff = forest_model.offset_                              # scikit-learn's cut-off (negative, like the scores)
cutoff_score = -raw_cutoff                                     # flip it, same as the scores
cutoff_path = -np.log2(cutoff_score) * normal_path             # turn it into a path length with the ruler

# 7. count the frames with a shorter path than the cut-off
# unusual_count = (average_paths < cutoff_path).sum()

# 8. sort the frames from biggest path to smallest (gives frame positions, not values)
ranking = np.argsort(average_paths)

# 9. where to save the report: app/reports/isolation_report_demo.txt
REPORTS_DIR = MODELS_DIR.parent / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)
report_path = REPORTS_DIR / f'isolation_report_{USER}.txt'

# 10. write every frame, from rank 1 (smallest path = most unusual) to the last (biggest path)
with open(report_path, 'w', encoding='utf-8') as report_file:
    report_file.write('rank  frame#  h_ratio  v_openness     yaw   pitch    roll  avg path  result\n')

    line_drawn = False                                   # the cut-off line hasn't been written yet
    for rank, frame_position in enumerate(ranking, start=1):
        frame_path = average_paths[frame_position]

        # write the cut-off line once, just before the first unusual frame
        if frame_path >= cutoff_path and not line_drawn:
            report_file.write(f'---------------------- cut-off line = {cutoff_path:.2f} ----------------------\n')
            line_drawn = True

        h_ratio, v_openness, yaw, pitch, roll = feature_table[frame_position]
        result = 'UNUSUAL' if frame_path < cutoff_path else 'normal'

        report_file.write(f'{rank:4}  {frame_position + 1:6}  {h_ratio:7.3f}  {v_openness:10.3f}  '
                          f'{yaw:6.2f}  {pitch:6.2f}  {roll:6.2f}  {frame_path:8.2f}  {result}\n')

print("Typical Cut for Each Tree: ", normal_path)

print('frame #1:  avg path', round(average_paths[0], 2))
print('frame #36: avg path', round(average_paths[35], 2))

print('cut-off path:', round(cutoff_path, 2))
# print('unusual frames:', unusual_count)


# print("raw score", raw_scores[0])
# print("flipped score", anomaly_scores[0])