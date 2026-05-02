#!/usr/bin/env bash
set -euo pipefail

if [ -d /root/autodl-tmp/deviant_detection_RNN ]; then
  cd /root/autodl-tmp/deviant_detection_RNN
  PY=python
  ROOT=/root/autodl-tmp/deviant_detection_RNN
  DEVICE=cuda
else
  cd "$(dirname "$0")"
  PY="$(pwd)/.venv/bin/python"
  if [ ! -x "$PY" ]; then
    PY=python3
  fi
  ROOT="$(pwd)"
  DEVICE=cpu
fi

# --- parse flags ---
SWEEP=false
SWEEP_SIGMA_OTHER="0,0.02,0.05,0.1,0.15,0.20,0.25,0.30"
SWEEP_P_OTHER="0.25,0.5,1.0"
SWEEP_SIGMA_SILENCE="0"
EXTRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --sweep) SWEEP=true ;;
    --sweep-sigma-other=*) SWEEP_SIGMA_OTHER="${arg#*=}" ;;
    --sweep-p-other=*) SWEEP_P_OTHER="${arg#*=}" ;;
    --sweep-sigma-silence=*) SWEEP_SIGMA_SILENCE="${arg#*=}" ;;
    *) EXTRA_ARGS+=("$arg") ;;
  esac
done

CFG_JSON="$ROOT/debug_stochastic_dynamic_preoffset_auxce_ablation_v1/recommended_fullish_config.json"
DYNAMIC_START=deviant_onset
AUX_W=0.5
AUX_START=deviant_onset
ANTI_TOKENS=20
ANTI_W=0.1

if [ -f "$CFG_JSON" ]; then
  CFG_TMP=$(mktemp)
  "$PY" -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get('dynamic_start', 'deviant_onset'))
print(cfg.get('aux_token_ce_weight', 0.5))
print(cfg.get('aux_token_ce_start', 'deviant_onset'))
print(cfg.get('anti_immediate_stop_tokens', 20))
print(cfg.get('anti_immediate_stop_weight', 0.1))
" "$CFG_JSON" > "$CFG_TMP" 2>&1 || true
  if [ -s "$CFG_TMP" ]; then
    readarray -t CFG_LINES < "$CFG_TMP"
    DYNAMIC_START="${CFG_LINES[0]}"
    AUX_W="${CFG_LINES[1]}"
    AUX_START="${CFG_LINES[2]}"
    ANTI_TOKENS="${CFG_LINES[3]}"
    ANTI_W="${CFG_LINES[4]}"
  fi
  rm -f "$CFG_TMP"
fi

ONLINE_SUP_START=deviant_start
if [ "$DYNAMIC_START" = "trial_start" ]; then
  ONLINE_SUP_START=trial_start
elif [ "$DYNAMIC_START" = "deviant_end" ]; then
  ONLINE_SUP_START=deviant_end
fi

BASE_SAVE_DIR="$ROOT/fullish_single_preoffset_auxce_minDiff25_v1"

# ============================================================
#  PHASE 1: Train 500ms base model (curriculum 100→300→500)
# ============================================================
if [ "$SWEEP" = false ]; then
  echo ""
  echo "========================================"
  echo "[fullish] PHASE 1: Train 500ms base model"
  echo "[fullish] curriculum 100→300→500, lr=1e-3, single-phase"
  echo "========================================"
  echo ""

  "$PY" train_online.py \
    --data_dir "$ROOT/gm_human_isi700" \
    --device "$DEVICE" \
    --seed 42 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 0.0 \
    --hidden_dim 512 \
    --num_layers 1 \
    --dropout 0.0 \
    --layer_norm \
    --chunk_len 1024 \
    --lambda_token 0.2 \
    --tone_ms 50 \
    --ramp_ms 5 \
    --token_ms 5 \
    --add_eos \
    --isi_schedule 100,300,500 \
    --epochs_per_isi 20 \
    --stage_perf_threshold 0.85 \
    --min_stage_epochs 8 \
    --force_full_last_stage \
    --sigma_other_noise 0.05 \
    --p_other_noise 1.0 \
    --sigma_silence_noise 0.0 \
    --stimuli_min_diff 25 \
    --block_context_training \
    --online_decision_training \
    --decision_policy hazard_bayesian \
    --use_stop_head \
    --decision_cost_mode stochastic_expected_cost \
    --use_hazard_prior \
    --hazard_prior_mode add_log_prior \
    --hazard_prior_weight 1.0 \
    --sampling_temperature 1.0 \
    --stop_temperature 1.0 \
    --online_loss_weight 0.5 \
    --online_ce_weight 0.1 \
    --time_cost_w 0.001 \
    --dynamic_start "$DYNAMIC_START" \
    --rt_logging_reference deviant_end \
    --cost_reference deviant_onset \
    --clamp_negative_cost_time \
    --aux_token_ce_weight "$AUX_W" \
    --aux_token_ce_start "$AUX_START" \
    --aux_token_ce_end trial_end \
    --anti_immediate_stop \
    --anti_immediate_stop_tokens "$ANTI_TOKENS" \
    --anti_immediate_stop_weight "$ANTI_W" \
    --online_supervision_start "$ONLINE_SUP_START" \
    --online_supervision_end trial_end \
    --save_dir "${BASE_SAVE_DIR}_500base" \
    "${EXTRA_ARGS[@]}"

  echo ""
  echo "[fullish] PHASE 1 complete. Base model saved to ${BASE_SAVE_DIR}_500base"
  echo "[fullish] To run noise sweep on 700ms:"
  echo "[fullish]   bash $0 --sweep"
  echo ""

# ============================================================
#  PHASE 2: Sweep noise on 700ms, initialized from 500 base
# ============================================================
else
  BASE_CKPT="${BASE_SAVE_DIR}_500base/best.pt"
  if [ ! -f "$BASE_CKPT" ]; then
    echo "[fullish] ERROR: Base checkpoint not found at $BASE_CKPT"
    echo "[fullish] Run without --sweep first to train the 500ms base model."
    exit 1
  fi

  echo ""
  echo "========================================"
  echo "[fullish] PHASE 2: Noise sweep on 700ms"
  echo "[fullish] init_from=$BASE_CKPT"
  echo "[fullish] sigma_other=$SWEEP_SIGMA_OTHER"
  echo "[fullish] p_other=$SWEEP_P_OTHER"
  echo "========================================"
  echo ""

  "$PY" train_online.py \
    --data_dir "$ROOT/gm_human_isi700" \
    --device "$DEVICE" \
    --seed 42 \
    --init_from "$BASE_CKPT" \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --hidden_dim 512 \
    --num_layers 1 \
    --dropout 0.0 \
    --layer_norm \
    --chunk_len 1024 \
    --lambda_token 0.2 \
    --tone_ms 50 \
    --ramp_ms 5 \
    --token_ms 5 \
    --add_eos \
    --isi_schedule 700 \
    --epochs_per_isi 15 \
    --stage_perf_threshold 0.0 \
    --min_stage_epochs 0 \
    --sigma_other_noise 0.05 \
    --p_other_noise 1.0 \
    --sigma_silence_noise 0.0 \
    --stimuli_min_diff 25 \
    --block_context_training \
    --online_decision_training \
    --decision_policy hazard_bayesian \
    --use_stop_head \
    --decision_cost_mode stochastic_expected_cost \
    --use_hazard_prior \
    --hazard_prior_mode add_log_prior \
    --hazard_prior_weight 1.0 \
    --sampling_temperature 1.0 \
    --stop_temperature 1.0 \
    --online_loss_weight 0.5 \
    --online_ce_weight 0.1 \
    --time_cost_w 0.001 \
    --dynamic_start "$DYNAMIC_START" \
    --rt_logging_reference deviant_end \
    --cost_reference deviant_onset \
    --clamp_negative_cost_time \
    --aux_token_ce_weight "$AUX_W" \
    --aux_token_ce_start "$AUX_START" \
    --aux_token_ce_end trial_end \
    --anti_immediate_stop \
    --anti_immediate_stop_tokens "$ANTI_TOKENS" \
    --anti_immediate_stop_weight "$ANTI_W" \
    --online_supervision_start "$ONLINE_SUP_START" \
    --online_supervision_end trial_end \
    --save_dir "${BASE_SAVE_DIR}_700sweep" \
    --sweep \
    --sweep_sigma_other "$SWEEP_SIGMA_OTHER" \
    --sweep_p_other "$SWEEP_P_OTHER" \
    --sweep_sigma_silence "$SWEEP_SIGMA_SILENCE" \
    "${EXTRA_ARGS[@]}"
fi
