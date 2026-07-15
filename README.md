# deviant_detection_RNN
MSC project reproducing fMRI data with RNN

## We are trying to reproduce the fMRI data from this paper:

Alejandro Tabas, Glad Mihai, Stefan Kiebel, Robert Trampel, Katharina von Kriegstein (2020) Abstract rules drive adaptation in the subcortical sensory pathway eLife 9:e64501

For now the point is to train the RNN to detect where the deviant is.

2025/11/14

## Code layout

The current training CLI remains `train_online.py`, but reusable pieces have been
split into smaller modules under `deviant_detection/`:

- `deviant_detection/stimuli.py`: stimulus block generation wrapper.
- `deviant_detection/data.py`: online/prerendered dataset classes.
- `deviant_detection/token_geometry.py`: deviant-position and token-window geometry.
- `deviant_detection/common.py`: shared IO, metrics, seed, and CLI helpers.

See `docs/github_code_list.md` for the recommended GitHub include list and the
files that should stay out of the cleaned repo.
