# Builds AEye_Isolation_Forest.ipynb (for Google Colab) from aeye_isolation_forest.py.
# It also pastes app/cheat/frame_features.py into the notebook, so Colab reads the
# features with exactly the app's code. Re-run after editing either file:
#     python build_notebook.py
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'aeye_isolation_forest.py'
FEATURES_PY = HERE.parent / 'app' / 'cheat' / 'frame_features.py'
OUT = HERE / 'AEye_Isolation_Forest.ipynb'
PLACEHOLDER = 'FRAME_FEATURES_SRC = None'


def lines(text):
    text = text.strip('\n')
    parts = text.split('\n')
    return [p + '\n' for p in parts[:-1]] + [parts[-1]]


def main():
    src = SOURCE.read_text(encoding='utf-8')
    feat = FEATURES_PY.read_text(encoding='utf-8')
    assert PLACEHOLDER in src, 'placeholder line missing from the source'
    src = src.replace(PLACEHOLDER, f'FRAME_FEATURES_SRC = {feat!r}', 1)

    cells = []
    for chunk in re.split(r'^# %%', src, flags=re.M)[1:]:
        header, _, body = chunk.partition('\n')
        if header.strip() == '[markdown]':
            md = '\n'.join(re.sub(r'^# ?', '', ln) for ln in body.strip('\n').split('\n'))
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': lines(md)})
        elif body.strip():
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': lines(body)})

    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'colab': {'provenance': [], 'name': OUT.name},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                       'language_info': {'name': 'python'}}}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {OUT.name}: {len(cells)} cells')


if __name__ == '__main__':
    main()
