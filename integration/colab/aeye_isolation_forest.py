# %% [markdown]
# # AEye - Isolation Forest (Google Colab)
#
# Trains an **Isolation Forest** on videos of **normal** exam behaviour, then
# **validates** it on videos with labelled cheating, with graphs.
#
# The forest uses the same 5 numbers per frame as the AEye app:
#
# | feature | what it is |
# |---|---|
# | `h_ratio` | iris left/right between the eye corners (0.5 = centre) |
# | `v_openness` | iris height above the eye corners (bigger = looking up) |
# | `yaw`, `pitch`, `roll` | head pose in degrees, from MediaPipe's face matrix |
#
# The feature code is copied from `integration/app/cheat/frame_features.py`, so the
# numbers here are exactly what the app computes live.
#
# ## 1. Put your videos in Google Drive
#
# ```
# MyDrive/AEye_IF/
#     normal/        <- videos with ONLY normal behaviour (used to learn "normal")
#     test/          <- videos to validate on (may contain cheating)
#     labels.csv     <- when the cheating happens in the test videos
# ```
#
# `labels.csv` (seconds from the start of each video; one row per cheating stretch):
#
# ```
# video,start_s,end_s
# 03_cheating_front,2,7
# 03_cheating_front,23,27
# ```
#
# `video` is the file name without `.mp4`. A test video with no rows counts as
# fully normal (useful for measuring false alarms).
#
# ## 2. Run all cells (Runtime -> Run all)
#
# At the end you get `isolation_general.npz`. Copy it into
# `integration/app/models/` and the app's Live Tracking and Video Test will use it.

# %%
# ---------------- settings (change these) ----------------
DRIVE_DIR = '/content/drive/MyDrive/AEye_IF'   # folder with normal/, test/, labels.csv
MODEL_NAME = 'general'      # app loads isolation_<student>.npz first, then isolation_general.npz

N_ESTIMATORS = 200          # number of trees
MAX_SAMPLES = 256           # frames per tree (256 = the original paper's default)
RANDOM_STATE = 42           # same random cuts every run

TARGET_FPR = 0.05           # cut-off: at most 5% of held-out NORMAL frames may be called unusual
VAL_FRACTION = 0.2          # share of the normal data held out for validation
ALERT_SECONDS = 2.0         # same 2-second rule as the app: unusual this long in a row = flag
FRAME_EVERY = 1             # 2 = keep every 2nd frame (faster training, same smoothing)

# %%
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mediapipe'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = Path(DRIVE_DIR)
    APP_CHEAT_DIR = None
else:
    # running on a PC: python aeye_isolation_forest.py <data folder>
    WORK = Path(os.environ.get('AEYE_IF_DIR') or (sys.argv[1] if len(sys.argv) > 1 else 'AEye_IF'))
    here = Path(globals().get('__file__', Path.cwd() / 'x')).resolve().parent
    APP_CHEAT_DIR = here.parent / 'app' / 'cheat'

NORMAL_DIR, TEST_DIR = WORK / 'normal', WORK / 'test'
FEAT_DIR, FIG_DIR = WORK / 'features', WORK / 'figures'
for d in (NORMAL_DIR, TEST_DIR, FEAT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('data folder:', WORK)

# %% [markdown]
# ### Feature code (copied from `app/cheat/frame_features.py` - edit it there, not here)

# %%
FRAME_FEATURES_SRC = None   # build_notebook.py pastes app/cheat/frame_features.py here
if FRAME_FEATURES_SRC:
    Path('frame_features.py').write_text(FRAME_FEATURES_SRC, encoding='utf-8')
    sys.path.insert(0, str(Path.cwd()))
else:
    sys.path.insert(0, str(APP_CHEAT_DIR))
import importlib
import frame_features as ff
importlib.reload(ff)
FEATURES = ff.FEATURES
print('features:', FEATURES)

# %%
import numpy as np
import pandas as pd
import matplotlib
if not IN_COLAB and 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)

# chart style: light surface, recessive grid, one fixed colour per meaning
NORMAL_C, CHEAT_C, VAL_C = '#2a78d6', '#eb6834', '#1baf7a'   # blue / orange / aqua
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
FLAG_C = '#e34948'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.color': INK2, 'ytick.color': INK2,
    'legend.frameon': False, 'lines.linewidth': 2, 'figure.dpi': 110, 'axes.axisbelow': True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{name}.png', dpi=150)
    plt.show() if IN_COLAB or 'ipykernel' in sys.modules else None
    plt.close(fig)

# %% [markdown]
# ## 3. Read the 5 features from every video
# (cached in `features/` - delete a CSV to re-read that video)

# %%
TASK = WORK / 'face_landmarker.task'
if not TASK.exists():
    url = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
           'face_landmarker/float16/1/face_landmarker.task')
    print('downloading', url)
    urllib.request.urlretrieve(url, TASK)

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def video_features(path, group):
    cache = FEAT_DIR / f'{group}__{path.stem}.csv'
    if cache.exists():
        return pd.read_csv(cache)
    start = time.time()
    rows = ff.extract_video_features(path, TASK, every=FRAME_EVERY,
                                     progress=lambda i, n: print(f'   {path.name}: frame {i}/{n}', end='\r'))
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f'   {path.name}: {len(df)} frames, {df.face.mean():.0%} with a face, '
          f'{time.time() - start:.0f}s')
    return df


frames = []
for group, folder in (('normal', NORMAL_DIR), ('test', TEST_DIR)):
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT):
        df = video_features(path, group)
        df['video'], df['group'] = path.stem, group
        frames.append(df)
if not frames:
    raise SystemExit(f'No videos found. Put normal videos in {NORMAL_DIR} and test videos in {TEST_DIR}.')
data = pd.concat(frames, ignore_index=True)
data = data[data.face.astype(bool)].dropna(subset=FEATURES).reset_index(drop=True)

# labels: 1 = cheating, 0 = normal
labels_path = WORK / 'labels.csv'
labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame(columns=['video', 'start_s', 'end_s'])
labels['video'] = labels['video'].astype(str).str.replace(r'\.\w+$', '', regex=True)
data['cheat'] = 0
for lab in labels.itertuples():
    inside = (data.video == lab.video) & (data.t >= lab.start_s) & (data.t <= lab.end_s)
    data.loc[inside, 'cheat'] = 1
if (data[data.group == 'normal'].cheat == 1).any():
    print('WARNING: labels.csv marks cheating inside a NORMAL video - those frames are dropped from training.')

summary = data.groupby(['group', 'video']).agg(frames=('t', 'size'), seconds=('t', 'max'),
                                               cheat_frames=('cheat', 'sum'))
print(summary.to_string())

# %% [markdown]
# ## 4. Split the normal data: train vs held-out validation
# Split **by video** when there are 3+ normal videos (so validation is on a
# video the forest never saw); otherwise the last 20% of each normal video is held out.

# %%
normal = data[(data.group == 'normal') & (data.cheat == 0)].copy()
normal_videos = sorted(normal.video.unique())
rng = np.random.default_rng(RANDOM_STATE)
if len(normal_videos) >= 3:
    n_val = max(1, round(len(normal_videos) * VAL_FRACTION))
    val_videos = {str(v) for v in rng.choice(normal_videos, n_val, replace=False)}
    normal['split'] = np.where(normal.video.isin(val_videos), 'val', 'train')
    # cross-validation groups: one fold per video
    normal['fold'] = normal.video
    print('validation videos:', sorted(val_videos))
else:
    cut = normal.groupby('video').t.transform(lambda s: s.quantile(1 - VAL_FRACTION))
    normal['split'] = np.where(normal.t > cut, 'val', 'train')
    # cross-validation groups: 5 time blocks per video
    normal['fold'] = normal.video + '#' + normal.groupby('video').t.transform(
        lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)).astype(str)
    print('fewer than 3 normal videos -> last', f'{VAL_FRACTION:.0%}', 'of each is validation')

train = normal[normal.split == 'train']
val = normal[normal.split == 'val']
test = data[data.group == 'test'].copy()
X_train = train[FEATURES].to_numpy(float)
print(f'train {len(train)} frames | validation {len(val)} normal frames | '
      f'test {len(test)} frames ({int(test.cheat.sum())} cheating)')
if len(train) < 200:
    raise SystemExit('Need at least ~200 normal frames with a face to train. Add more normal video.')

# %% [markdown]
# ## 5. Train the Isolation Forest and pick the cut-off
# Anomaly score = `-score_samples` (about 0.3 = very normal, up to 1 = very unusual).
# The cut-off is set so only `TARGET_FPR` of the **held-out normal** frames are
# above it - it is never tuned on the cheating labels.

# %%
def fit_forest(X, seed=RANDOM_STATE):
    return IsolationForest(n_estimators=N_ESTIMATORS, max_samples=min(MAX_SAMPLES, len(X)),
                           random_state=seed).fit(X)


def anomaly(forest, df):
    return -forest.score_samples(df[FEATURES].to_numpy(float))


forest = fit_forest(X_train)
train_s, val_s = anomaly(forest, train), anomaly(forest, val)
threshold = float(np.quantile(val_s, 1 - TARGET_FPR))
threshold_percentile = float((train_s <= threshold).mean() * 100)
test['score'] = anomaly(forest, test) if len(test) else []
test['pred'] = (test.score > threshold).astype(int)
print(f'cut-off score = {threshold:.4f}  (= {threshold_percentile:.1f}th percentile of training scores)')
print(f'unusual frames: train {np.mean(train_s > threshold):.1%}, held-out normal {np.mean(val_s > threshold):.1%}')

# %% [markdown]
# ## 6. Validation numbers
# **Frame level** - every test frame vs its label.
# **Event level** - the app's real behaviour: a flag needs `ALERT_SECONDS` of unusual
# frames in a row. A cheating stretch counts as *caught* if a flag overlaps it.

# %%
def flag_intervals(df, alert=ALERT_SECONDS):
    """Same 2-second rule as front_cam_worker.py: returns [(start_s, end_s), ...]."""
    out, since, active = [], None, False
    for t, unusual in zip(df.t.to_numpy(), df.pred.to_numpy()):
        if unusual:
            since = t if since is None else since
            if t - since >= alert:
                if not active:
                    out.append([since, t])
                    active = True
                out[-1][1] = t
        else:
            since, active = None, False
    return out


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


results = {'n_train_frames': int(len(train)), 'n_val_frames': int(len(val)),
           'n_test_frames': int(len(test)), 'threshold': threshold,
           'val_normal_fpr': float(np.mean(val_s > threshold))}
has_labels = len(test) and test.cheat.nunique() == 2

if has_labels:
    y, s, p = test.cheat.to_numpy(), test.score.to_numpy(), test.pred.to_numpy()
    results.update(
        roc_auc=float(roc_auc_score(y, s)), average_precision=float(average_precision_score(y, s)),
        accuracy=float(accuracy_score(y, p)), precision=float(precision_score(y, p, zero_division=0)),
        recall=float(recall_score(y, p, zero_division=0)), f1=float(f1_score(y, p, zero_division=0)))

events = []           # one row per true cheating stretch
false_alarms, normal_minutes, flags_by_video = 0, 0.0, {}
for video, df in test.groupby('video'):
    df = df.sort_values('t')
    flags = flag_intervals(df)
    flags_by_video[video] = flags
    truth = [(r.start_s, r.end_s) for r in labels[labels.video == video].itertuples()]
    for tr in truth:
        hits = [f for f in flags if overlaps(f, tr)]
        events.append({'video': video, 'start_s': tr[0], 'end_s': tr[1], 'caught': bool(hits),
                       'delay_s': max(0.0, hits[0][0] + ALERT_SECONDS - tr[0]) if hits else None})
    false_alarms += sum(not any(overlaps(f, tr) for tr in truth) for f in flags)
    normal_minutes += (df.cheat == 0).sum() / max(1, len(df)) * (df.t.max() - df.t.min()) / 60
events = pd.DataFrame(events)
if len(events):
    results.update(events_total=int(len(events)), events_caught=int(events.caught.sum()),
                   event_recall=float(events.caught.mean()),
                   mean_delay_s=float(events.delay_s.dropna().mean()) if events.caught.any() else None)
results.update(false_alarm_flags=int(false_alarms),
               false_alarms_per_min=float(false_alarms / normal_minutes) if normal_minutes else None)

print(json.dumps(results, indent=2))
if len(events):
    print(events.to_string(index=False))
if not has_labels:
    print('\nNo labelled cheating in test/ - only false-alarm numbers are available.')

# %% [markdown]
# ## 7. Graphs
# ### 7a. Do cheating frames look different? (feature distributions)

# %%
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
for ax, f in zip(axes, FEATURES):
    groups = [('train normal', train[f], NORMAL_C)]
    if has_labels:
        groups += [('test normal', test[test.cheat == 0][f], VAL_C),
                   ('test cheating', test[test.cheat == 1][f], CHEAT_C)]
    lo, hi = np.nanpercentile(pd.concat([g[1] for g in groups]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 50)
    for name, vals, c in groups:
        ax.hist(vals, bins=bins, density=True, histtype='step', linewidth=2, color=c, label=name)
    ax.set_title(f)
    ax.set_yticks([])
axes[0].legend(loc='upper left', fontsize=8)
fig.suptitle('Feature distributions: normal vs cheating', color=INK, fontweight='bold')
save(fig, '1_feature_distributions')

# %% [markdown]
# ### 7b. Anomaly scores - cheating frames should sit to the right of the cut-off

# %%
fig, ax = plt.subplots(figsize=(9, 4))
all_s = np.concatenate([train_s, val_s] + ([test.score.to_numpy()] if len(test) else []))
bins = np.linspace(all_s.min(), all_s.max(), 60)
ax.hist(train_s, bins, density=True, histtype='step', color=NORMAL_C, label='train normal')
ax.hist(val_s, bins, density=True, histtype='step', color=VAL_C, label='held-out normal')
if has_labels:
    ax.hist(test[test.cheat == 1].score, bins, density=True, histtype='step', color=CHEAT_C,
            label='test cheating')
ax.axvline(threshold, color=INK, linestyle='--', linewidth=1.5)
ax.annotate(f'cut-off {threshold:.3f}', (threshold, ax.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9)
ax.set_xlabel('anomaly score (higher = more unusual)')
ax.set_ylabel('share of frames')
ax.set_yticks([])
ax.legend()
ax.set_title('Anomaly score distribution')
save(fig, '2_score_distribution')

# %% [markdown]
# ### 7c. ROC and precision-recall curves (frame level)

# %%
if has_labels:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fpr, tpr, _ = roc_curve(y, s)
    a1.plot(fpr, tpr, color=NORMAL_C, label=f"Isolation Forest (AUC = {results['roc_auc']:.3f})")
    a1.plot([0, 1], [0, 1], color=INK2, linestyle=':', linewidth=1, label='random guess (0.5)')
    tp_rate = results['recall']
    fp_rate = float(((p == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))
    a1.plot([fp_rate], [tp_rate], 'o', color=INK, markersize=8)
    a1.annotate('chosen cut-off', (fp_rate, tp_rate), xytext=(8, -14), textcoords='offset points',
                fontsize=9, color=INK)
    a1.set(xlabel='false positive rate', ylabel='true positive rate', title='ROC curve',
           xlim=(0, 1), ylim=(0, 1.01))
    a1.legend(loc='lower right')
    prec, rec, _ = precision_recall_curve(y, s)
    a2.plot(rec, prec, color=CHEAT_C, label=f"AP = {results['average_precision']:.3f}")
    a2.axhline(y.mean(), color=INK2, linestyle=':', linewidth=1, label=f'random guess ({y.mean():.2f})')
    a2.plot([results['recall']], [results['precision']], 'o', color=INK, markersize=8)
    a2.set(xlabel='recall', ylabel='precision', title='Precision-recall curve',
           xlim=(0, 1), ylim=(0, 1.01))
    a2.legend(loc='upper right')
    save(fig, '3_roc_pr')

# %% [markdown]
# ### 7d. Confusion matrix (frame level) and caught cheating stretches (event level)

# %%
if has_labels:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1, 1.3]})
    cm = confusion_matrix(y, p, labels=[0, 1])
    a1.imshow(cm, cmap='Blues')
    a1.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        a1.text(j, i, f'{v}\n({v / cm[i].sum():.0%} of row)', ha='center', va='center',
                color='white' if v > cm.max() / 2 else INK, fontsize=10)
    a1.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['said normal', 'said unusual'],
           yticklabels=['really normal', 'really cheating'], title='Frames')
    if len(events):
        caught = int(events.caught.sum())
        a2.barh(['cheating stretches'], [caught], color=NORMAL_C, label='caught')
        a2.barh(['cheating stretches'], [len(events) - caught], left=[caught], color=CHEAT_C,
                label='missed')
        a2.barh(['false-alarm flags'], [false_alarms], color=INK2, label='flag with no cheating')
        a2.text(len(events), 0, f'  {caught} of {len(events)} caught', va='center', color=INK)
        a2.text(false_alarms, 1, f'  {false_alarms} false alarms', va='center', color=INK)
        a2.set_xlim(0, max(len(events), false_alarms) * 1.45 + 1)
        a2.set_title(f'Events with the {ALERT_SECONDS:g}-second rule')
        a2.set_xlabel('count')
        a2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.38), ncol=3)
    save(fig, '4_confusion_events')

# %% [markdown]
# ### 7e. Timeline of each test video
# Score over time, the labelled cheating (shaded), and the flags the app would raise.

# %%
test_videos = sorted(test.video.unique())
if test_videos:
    fig, axes = plt.subplots(len(test_videos), 1, figsize=(13, 2.5 * len(test_videos)), squeeze=False)
    for ax, video in zip(axes[:, 0], test_videos):
        df = test[test.video == video].sort_values('t')
        for k, lab in enumerate(labels[labels.video == video].itertuples()):
            ax.axvspan(lab.start_s, lab.end_s, color=CHEAT_C, alpha=0.18, linewidth=0,
                       label='labelled cheating' if k == 0 else None)
        ax.plot(df.t, df.score, color=NORMAL_C, linewidth=1.2, label='anomaly score')
        ax.axhline(threshold, color=INK, linestyle='--', linewidth=1, label='cut-off')
        top = df.score.max() if len(df) else 1
        for k, (a, b) in enumerate(flags_by_video.get(video, [])):
            ax.plot([a, b], [top * 1.03] * 2, color=FLAG_C, linewidth=5, solid_capstyle='butt',
                    label='app flag (2 s rule)' if k == 0 else None)
        ax.set_title(video, loc='left', fontsize=10)
        ax.set_ylabel('score')
    axes[-1, 0].set_xlabel('video time (s)')
    axes[0, 0].legend(loc='lower right', bbox_to_anchor=(1, 1.02), ncol=4, fontsize=8)
    save(fig, '5_timelines')

# %% [markdown]
# ### 7f. Where should the cut-off be? (threshold sweep)
# Moving the cut-off trades false alarms on normal frames against catching cheating.

# %%
pcts = np.linspace(50, 99.9, 120)
cuts = np.percentile(val_s, pcts)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(100 - pcts, [np.mean(val_s > c) for c in cuts], color=VAL_C,
        label='held-out normal flagged (false alarms)')
if has_labels:
    ax.plot(100 - pcts, [recall_score(y, s > c, zero_division=0) for c in cuts], color=CHEAT_C,
            label='cheating frames caught (recall)')
    ax.plot(100 - pcts, [f1_score(y, s > c, zero_division=0) for c in cuts], color=NORMAL_C, label='F1')
ax.axvline(TARGET_FPR * 100, color=INK, linestyle='--', linewidth=1)
ax.annotate(f'chosen ({TARGET_FPR:.0%})', (TARGET_FPR * 100, 0.95), xytext=(6, 0),
            textcoords='offset points', fontsize=9, color=INK)
ax.set(xlabel='target false-positive rate on held-out normal (%)', ylabel='rate', ylim=(0, 1.02),
       title='Threshold sweep')
ax.legend(loc='center right')
save(fig, '6_threshold_sweep')

# %% [markdown]
# ### 7g. What the forest learned as "normal" (2-D slices)
# Colour = anomaly score with the other 3 features held at their normal median.
# Dots = test frames (blue normal, orange cheating).

# %%
med = train[FEATURES].median()
pairs = [('yaw', 'pitch'), ('h_ratio', 'v_openness')]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (fx, fy) in zip(axes, pairs):
    ref = pd.concat([train, test]) if len(test) else train
    xr = np.nanpercentile(ref[fx], [0, 100]); yr = np.nanpercentile(ref[fy], [0, 100])
    pad_x, pad_y = (xr[1] - xr[0]) * 0.1, (yr[1] - yr[0]) * 0.1
    gx, gy = np.meshgrid(np.linspace(xr[0] - pad_x, xr[1] + pad_x, 120),
                         np.linspace(yr[0] - pad_y, yr[1] + pad_y, 120))
    grid = pd.DataFrame({f: np.full(gx.size, med[f]) for f in FEATURES})
    grid[fx], grid[fy] = gx.ravel(), gy.ravel()
    z = anomaly(forest, grid).reshape(gx.shape)
    cf = ax.contourf(gx, gy, z, levels=20, cmap='Greys', alpha=0.55)
    ax.contour(gx, gy, z, levels=[threshold], colors=[INK], linewidths=1.5, linestyles='--')
    sample = train.sample(min(1500, len(train)), random_state=0)
    ax.scatter(sample[fx], sample[fy], s=4, color=NORMAL_C, alpha=0.35, label='train normal')
    if has_labels:
        ch = test[test.cheat == 1]
        ch = ch.sample(min(1500, len(ch)), random_state=0)
        ax.scatter(ch[fx], ch[fy], s=8, color=CHEAT_C, alpha=0.6, label='test cheating')
    ax.set(xlabel=fx, ylabel=fy, title=f'{fx} vs {fy}')
    ax.grid(False)
    fig.colorbar(cf, ax=ax, label='anomaly score')
axes[0].legend(loc='upper right', markerscale=3)
save(fig, '7_decision_maps')

# %% [markdown]
# ### 7h. Which features matter? (permutation importance)
# Shuffle one feature at a time in the test set; the bigger the AUC drop, the
# more the forest relies on it.

# %%
if has_labels:
    base = results['roc_auc']
    drops = {}
    for f in FEATURES:
        vals = []
        for r in range(5):
            shuffled = test[FEATURES].copy()
            shuffled[f] = np.random.default_rng(r).permutation(shuffled[f].to_numpy())
            vals.append(base - roc_auc_score(y, anomaly(forest, shuffled)))
        drops[f] = (np.mean(vals), np.std(vals))
    order = sorted(drops, key=lambda f: drops[f][0])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(order, [drops[f][0] for f in order], xerr=[drops[f][1] for f in order],
            color=NORMAL_C, height=0.55, error_kw={'ecolor': INK2})
    for i, f in enumerate(order):
        ax.text(max(drops[f][0], 0) + drops[f][1] + 0.003, i, f'{drops[f][0]:+.3f}', va='center', fontsize=9, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set(xlabel='drop in ROC AUC when the feature is shuffled', title='Feature importance')
    results['permutation_importance'] = {f: float(drops[f][0]) for f in FEATURES}
    save(fig, '8_feature_importance')

# %% [markdown]
# ## 8. Is it stable? (cross-validation + learning curve)
# **Cross-validation:** leave one normal video (or time block) out, train on the
# rest, and measure how much of the left-out normal data gets called unusual - plus
# the test AUC. Similar numbers across folds = the forest generalises.
# **Learning curve:** does more normal video help?

# %%
cv_rows = []
for fold in sorted(normal.fold.unique()):
    tr, ho = normal[normal.fold != fold], normal[normal.fold == fold]
    if len(tr) < 200 or len(ho) < 20:
        continue
    fm = fit_forest(tr[FEATURES].to_numpy(float))
    tr_s = anomaly(fm, tr)
    cut = np.percentile(tr_s, threshold_percentile)   # same rule the app uses
    row = {'held_out': fold, 'frames': len(ho), 'normal_flagged': float(np.mean(anomaly(fm, ho) > cut))}
    if has_labels:
        row['test_auc'] = float(roc_auc_score(y, anomaly(fm, test)))
    cv_rows.append(row)
cv = pd.DataFrame(cv_rows)
if len(cv):
    print(cv.to_string(index=False))
    results['cross_validation'] = {
        'folds': int(len(cv)),
        'normal_flagged_mean': float(cv.normal_flagged.mean()),
        'normal_flagged_std': float(cv.normal_flagged.std()),
        **({'test_auc_mean': float(cv.test_auc.mean()), 'test_auc_std': float(cv.test_auc.std())}
           if has_labels else {})}

sizes = [f for f in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0) if f * len(train) >= 100]
lc = []
for frac in sizes:
    for seed in range(3):
        sub = train.sample(frac=frac, random_state=seed)
        fm = fit_forest(sub[FEATURES].to_numpy(float), seed)
        cut = np.percentile(anomaly(fm, sub), threshold_percentile)
        lc.append({'frames': len(sub), 'val_fpr': float(np.mean(anomaly(fm, val) > cut)),
                   'auc': float(roc_auc_score(y, anomaly(fm, test))) if has_labels else np.nan})
lc = pd.DataFrame(lc).groupby('frames').agg(['mean', 'std']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if len(cv):
    xs = np.arange(len(cv))
    axes[0].bar(xs, cv.normal_flagged, color=VAL_C, width=0.6, label='left-out normal flagged')
    axes[0].axhline(TARGET_FPR, color=INK, linestyle='--', linewidth=1, label=f'target {TARGET_FPR:.0%}')
    axes[0].set_xticks(xs, cv.held_out, rotation=45, ha='right', fontsize=8)
    axes[0].set(ylabel='share of frames', title='Cross-validation: false alarms per fold')
    axes[0].legend()
    if has_labels:
        title = f"test AUC per fold: {cv.test_auc.mean():.3f} ± {cv.test_auc.std():.3f}"
        axes[0].set_title('Cross-validation: false alarms per fold\n' + title, fontsize=11)
ax = axes[1]
ax.errorbar(lc.frames, lc.val_fpr['mean'], yerr=lc.val_fpr['std'], color=VAL_C, marker='o',
            label='held-out normal flagged')
if has_labels:
    ax.errorbar(lc.frames, lc.auc['mean'], yerr=lc.auc['std'], color=NORMAL_C, marker='o',
                label='test ROC AUC')
ax.set(xscale='log', xlabel='normal training frames', ylabel='value', ylim=(0, 1.02),
       title='Learning curve')
ax.set_xticks(lc.frames, [f'{n:,}' for n in lc.frames], fontsize=8)
ax.minorticks_off()
ax.legend()
save(fig, '9_cross_validation_learning_curve')

# %% [markdown]
# ## 9. Export the model for the app
# Saves `isolation_<MODEL_NAME>.npz` (training frames + settings + cut-off) and
# `report_<MODEL_NAME>.json` (all the validation numbers). The app re-builds the
# forest from the training frames, so Colab's library versions don't matter.

# %%
import sklearn
meta = {
    'features': FEATURES, 'n_estimators': N_ESTIMATORS, 'max_samples': min(MAX_SAMPLES, len(X_train)),
    'random_state': RANDOM_STATE, 'threshold_percentile': threshold_percentile,
    'threshold_colab': threshold, 'target_fpr': TARGET_FPR, 'alert_seconds': ALERT_SECONDS,
    'normal_videos': normal_videos, 'test_videos': test_videos,
    'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'sklearn_version': sklearn.__version__,
    'validation': results,
}
out = WORK / f'isolation_{MODEL_NAME}.npz'
np.savez_compressed(out, X=X_train, meta=np.array(json.dumps(meta)))
(WORK / f'report_{MODEL_NAME}.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
print('saved', out, f'({out.stat().st_size / 1024:.0f} KB)')
print('figures in', FIG_DIR)
print(f'\nCopy {out.name} into integration/app/models/ to use it in the app.')
if IN_COLAB:
    from google.colab import files
    files.download(str(out))
