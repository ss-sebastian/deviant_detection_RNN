# deviant_detection_RNN

RNN modelling code for Sebastian's MSc thesis on auditory deviant detection and
human reaction-time behaviour.

The project trains recurrent neural networks on an 8-tone auditory deviant
detection task and evaluates whether model-derived reaction-time readouts explain
human RT structure beyond behavioural baselines.

The thesis/manuscript PDF is included here for context:

- `docs/msc_thesis_Sebastian.pdf`

## Scientific Context

This project is motivated by:

Alejandro Tabas, Glad Mihai, Stefan Kiebel, Robert Trampel, Katharina von
Kriegstein (2020), "Abstract rules drive adaptation in the subcortical sensory
pathway", eLife 9:e64501.

In this codebase, the model is trained to infer the deviant position in a tone
sequence. Behavioural evaluation then tests whether model decision dynamics align
with human RTs.

## Training Code

Use `train_online.py` as the main training entry point. It remains the
backwards-compatible CLI used for the experiments.

Core training files:

- `train_online.py`: main training CLI, loss definitions, curriculum logic, and training diagnostics.
- `model.py`: GRU model definition and token/readout heads.
- `gm_stimuli.py`: symbolic auditory stimulus generation.
- `stimulus_encoding.py`: token-level stimulus encoding, including Gaussian RF, EOS/BOS markers, and input noise.
- `rt_readout.py`: reaction-time readout utilities used during training diagnostics and evaluation.

Reusable training modules:

- `deviant_detection/stimuli.py`: stimulus block generation wrapper.
- `deviant_detection/data.py`: online/prerendered dataset classes.
- `deviant_detection/token_geometry.py`: deviant-position and token-window geometry.
- `deviant_detection/common.py`: shared IO, metrics, seed, AMP, and CLI helpers.

Typical training flow:

1. Generate or load stimulus blocks.
2. Encode tone sequences into token-level input features.
3. Train the GRU to predict deviant position using token/window/end losses.
4. Save checkpoints for behavioural model selection.

## Evaluation And Model-Selection Code

Use `eval_models_on_human.py` as the main behavioural evaluation entry point.
It evaluates trained checkpoints against human RT data without retraining.

Core evaluation files:

- `eval_models_on_human.py`: main model-on-human behavioural evaluation script.
- `rt_readout.py`: simple-threshold, Bayesian/cost, and accumulator-style RT readouts.
- `run_lopo_checkpoint_selection_eval.py`: leave-one-participant-out checkpoint selection with repeated noise seeds.
- `run_crossvalidated_merged_participant_train_loss.py`: cross-validated checkpoint/model-selection helper.
- `run_crossvalidated_behavioral_checkpoint_analysis.py`: checkpoint-level behavioural analysis helper.
- `run_crossvalidated_behavioral_checkpoint_analysis_multisource.py`: multisource extension of the checkpoint-analysis helper.

Typical evaluation flow:

1. Evaluate each checkpoint on human trial sequences.
2. Compute participant-level model metrics, including standalone and nested/incremental R2.
3. Select checkpoints using validation participants only.
4. Report held-out participant performance under LOPO selection.

## Repository Notes

The recommended upload list is documented in:

- `docs/github_code_list.md`

Generated outputs should not be committed by default. This includes checkpoints,
rendered datasets, run folders, temporary analysis outputs, figures, caches, and
large archives.

The repository intentionally does not include standalone manuscript plotting
scripts in the core upload list. Those are analysis-output scripts rather than
training/evaluation pipeline code.
