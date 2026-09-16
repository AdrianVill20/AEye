# Make the student's screen area from their calibration.
# Run by hand from integration/app:  ../.venv/Scripts/python.exe -m cheat.train_cheat_model --user ichoy

import argparse
import json
from statistics import median

from cheat.calibration_store import load_sessions
from cheat.cheat_detector import CheatDetector, MARGIN, graham_scan, grow, user_model_path


def head_k(samples, eye, head):
    # How much the eyes move when the head turns.
    groups = {}
    for run, s in samples:
        if s.get('head') and s.get('target'):
            groups.setdefault((run, tuple(s['target'])), []).append(s)
    top = bottom = 0.0
    for g in groups.values():
        eye_mean = sum(s[eye] for s in g) / len(g)
        head_mean = sum(s[head] for s in g) / len(g)
        top += sum((s[eye] - eye_mean) * (s[head] - head_mean) for s in g)
        bottom += sum((s[head] - head_mean) ** 2 for s in g)
    return -top / bottom if bottom else 0.0


def dot_points(session, area):
    # One point per dot (middle value of its frames).
    dots = {}
    for s in session['samples']:
        if s.get('target') and not s.get('head') and not s.get('val'):
            dots.setdefault(tuple(s['target']), []).append(
                area.point(s['h_ratio'], s['v_openness'], s['yaw'], s['pitch']))
    return [(median(p[0] for p in d), median(p[1] for p in d))
            for d in dots.values() if len(d) >= 10]


def newest_dots(sessions, area):
    # Dot points from the newest good calibration.
    for session in reversed(sessions):
        points = dot_points(session, area)
        if len(points) >= 3:
            return points, session
    return [], None


def train(user):
    # Make and save the screen area.
    sessions = load_sessions(user)
    if not sessions:
        raise FileNotFoundError(f'No dot calibration saved for "{user}".')

    # head fix uses all calibrations
    samples = [(run, s) for run, session in enumerate(sessions) for s in session['samples']]
    area = CheatDetector(kx=head_k(samples, 'h_ratio', 'yaw'),
                         ky=head_k(samples, 'v_openness', 'pitch'))

    # screen area uses the newest calibration
    points, session = newest_dots(sessions, area)
    if not points:
        raise ValueError(f'No usable dot calibration for "{user}" - calibrate again, '
                         f'keeping your face in view.')

    # graham scan, then make it a bit bigger
    area.hull = grow(graham_scan(points), MARGIN)
    out = user_model_path(user)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'user': user, 'hull': [list(p) for p in area.hull],
                   'kx': area.kx, 'ky': area.ky}, f)

    # check: test dots should be inside
    val = [s for s in session['samples'] if s.get('val')]
    inside = sum(not area.outside(s['h_ratio'], s['v_openness'], s['yaw'], s['pitch']) for s in val)
    return {'dots': len(points), 'corners': len(area.hull), 'runs': len(sessions),
            'val_inside': inside, 'val_total': len(val), 'kx': area.kx, 'ky': area.ky,
            'model_path': str(out)}


def main():
    # Train from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True,
                    help='student id (matches the calibration JSON / login)')
    args = ap.parse_args()

    try:
        result = train(args.user)
    except FileNotFoundError as exc:
        raise SystemExit(f'[TRAIN] {exc} Run calibration first.')
    except ValueError as exc:
        raise SystemExit(f'[TRAIN] {exc}')

    pct = 100 * result['val_inside'] / result['val_total'] if result['val_total'] else 0
    print(f"[TRAIN] Screen area from {result['dots']} dots of the newest run: "
          f"{result['corners']} corners. Head correction from {result['runs']} run(s): "
          f"kx={result['kx']:.4f} ky={result['ky']:.4f}")
    print(f"[TRAIN] Validation frames inside the area: "
          f"{result['val_inside']}/{result['val_total']} ({pct:.0f}%).")
    print(f"[TRAIN] Saved -> {result['model_path']}")


if __name__ == '__main__':
    main()
