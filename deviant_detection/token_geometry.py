from __future__ import annotations

from typing import Tuple

import torch


def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    """Map deviant position labels {4, 5, 6} to class indices {0, 1, 2}."""
    return (y_456 - 4).long()


def supervision_num_classes(supervision_mode: str) -> int:
    return 3


def tone_onset_token_in_trial(
    tone_pos_1idx: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    tone_idx0 = (tone_pos_1idx.long() - 1).clamp(min=0)
    step = int(tone_T + isi_T)
    return tone_idx0 * step


def tone_offset_token_in_trial(
    tone_pos_1idx: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_onset_token_in_trial(tone_pos_1idx, tone_T=tone_T, isi_T=isi_T) + int(tone_T) - 1


def deviant_end_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()
    step = int(tone_T + isi_T)
    return dev_idx * step + int(tone_T) - 1


def deviant_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()
    step = int(tone_T + isi_T)
    return dev_idx * step


def trial_end_token_in_trial(
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
) -> torch.Tensor:
    return torch.full_like(y_pos_456.long(), int(trial_T_tokens) - 1)


def next_standard_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    dev_idx = (y_pos_456 - 1).long()
    step = int(tone_T + isi_T)
    return (dev_idx + 1) * step


def previous_standard_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_onset_token_in_trial(y_pos_456.long() - 1, tone_T=tone_T, isi_T=isi_T)


def previous_standard_offset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_offset_token_in_trial(y_pos_456.long() - 1, tone_T=tone_T, isi_T=isi_T)


def first_possible_deviant_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return torch.zeros_like(y_pos_456.long(), dtype=torch.long) + int(3 * (tone_T + isi_T))


def next_tone_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_onset_token_in_trial(y_pos_456.long() + 1, tone_T=tone_T, isi_T=isi_T)


def next_tone_offset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_offset_token_in_trial(y_pos_456.long() + 1, tone_T=tone_T, isi_T=isi_T)


def second_next_tone_onset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_onset_token_in_trial(y_pos_456.long() + 2, tone_T=tone_T, isi_T=isi_T)


def second_next_tone_offset_token_in_trial(
    y_pos_456: torch.Tensor,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    return tone_offset_token_in_trial(y_pos_456.long() + 2, tone_T=tone_T, isi_T=isi_T)


def make_strict_online_p4_mask(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    """Strict-online mask: fixed P4 onset through trial end for every trial."""
    trial_id = (abs_t // int(trial_T_tokens)).long()
    within = (abs_t % int(trial_T_tokens)).long()
    p4_tok = first_possible_deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]
    mask = within.unsqueeze(0) >= p4_tok
    if mask.any() and torch.any(mask & (within.unsqueeze(0) < p4_tok)):
        raise RuntimeError("strict_online_p4 loss mask is active before P4")
    return mask


def make_strict_p4_post_offset_mask(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    """Causal strict-P4 mask: activate only after tone 4 has finished."""
    trial_id = (abs_t // int(trial_T_tokens)).long()
    within = (abs_t % int(trial_T_tokens)).long()
    p4_off_tok = tone_offset_token_in_trial(
        torch.full_like(y_pos_456.long(), 4),
        tone_T=tone_T,
        isi_T=isi_T,
    )[:, trial_id]
    mask = within.unsqueeze(0) > p4_off_tok
    if mask.any() and torch.any(mask & (within.unsqueeze(0) <= p4_off_tok)):
        raise RuntimeError("strict_p4_causal_ce loss mask is active before P4 outcome is known")
    return mask


def make_strict_p4_causal_target_probs(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
) -> torch.Tensor:
    """
    Causal 3-class targets for the strict P4 window.

    Before the true deviant, the target only uses information available so far.
    """
    trial_id = (abs_t // int(trial_T_tokens)).long()
    within = (abs_t % int(trial_T_tokens)).long().unsqueeze(0)
    y_cls = labels_to_class_index(y_pos_456)
    target_cls = y_cls[:, trial_id]

    probs = torch.zeros(
        target_cls.shape + (3,),
        dtype=torch.float32,
        device=abs_t.device,
    )

    p5_tok = tone_onset_token_in_trial(
        torch.full_like(y_pos_456.long(), 5),
        tone_T=tone_T,
        isi_T=isi_T,
    )[:, trial_id]
    p6_tok = tone_onset_token_in_trial(
        torch.full_like(y_pos_456.long(), 6),
        tone_T=tone_T,
        isi_T=isi_T,
    )[:, trial_id]
    dev_tok = deviant_onset_token_in_trial(y_pos_456, tone_T=tone_T, isi_T=isi_T)[:, trial_id]

    after_dev = within >= dev_tok
    probs.scatter_(-1, target_cls.unsqueeze(-1), after_dev.float().unsqueeze(-1))

    before_dev = ~after_dev
    y_is_5_or_6 = target_cls >= 1
    before_p5 = within < p5_tok
    between_p5_p6 = (within >= p5_tok) & (within < p6_tok)

    uncertain_5_or_6 = before_dev & y_is_5_or_6 & before_p5
    probs[..., 1] = torch.where(uncertain_5_or_6, probs[..., 1] + 0.5, probs[..., 1])
    probs[..., 2] = torch.where(uncertain_5_or_6, probs[..., 2] + 0.5, probs[..., 2])

    must_be_6 = before_dev & (target_cls == 2) & between_p5_p6
    probs[..., 2] = torch.where(must_be_6, torch.ones_like(probs[..., 2]), probs[..., 2])

    maybe_4 = before_dev & (target_cls == 0)
    probs[..., 0] = torch.where(maybe_4, torch.ones_like(probs[..., 0]), probs[..., 0])
    return probs


def make_tone_event_targets_tokens(
    abs_t: torch.Tensor,
    y_pos_456: torch.Tensor,
    trial_T_tokens: int,
    tone_T: int,
    isi_T: int,
    n_tones: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Binary targets for whether the currently heard tone is deviant."""
    trial_id = (abs_t // int(trial_T_tokens)).long()
    within = (abs_t % int(trial_T_tokens)).long()
    step = int(tone_T) + int(isi_T)
    tone_idx0 = within // int(step)
    in_tone = (tone_idx0 >= 0) & (tone_idx0 < int(n_tones)) & ((within % int(step)) < int(tone_T))

    tone_pos_1idx = (tone_idx0 + 1).view(1, -1)
    target = (tone_pos_1idx == y_pos_456[:, trial_id].long()).float()
    mask = in_tone.view(1, -1).expand_as(target)
    return mask, target


def infer_end_indices_from_T(
    T: int,
    trials_per_block: int = 10,
    end_offset_from_trial_end: int = 0,
) -> torch.Tensor:
    if T % trials_per_block != 0:
        raise ValueError(f"Cannot infer trial length: T={T} not divisible by {trials_per_block}")
    trial_T = T // trials_per_block
    end_offset = max(0, int(end_offset_from_trial_end))
    return torch.tensor([(i + 1) * trial_T - 1 - end_offset for i in range(trials_per_block)], dtype=torch.long)

