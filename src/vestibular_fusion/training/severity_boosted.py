"""Continuous-score training and diagnostics for two-stage severity residuals."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.anchor_pilot import metrics_from_rows
from ..model.severity_dynamics import NormalizedFractionalDoGKAN
from .token_pilot import cpu_state, token_inputs


def make_optimizer_step(score_mean, score_std):
    mean, std = float(score_mean), float(score_std)

    def step(head, optimizer, tokens, batch, device, protocol, positive_weight):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = [head.forward_outputs(token_inputs(tokens, [example], head.task, device))
                   for example in batch]
        logits = torch.cat([value["binary_logit"] for value in outputs])
        scores = torch.cat([value["continuous_score"] for value in outputs])
        labels = torch.tensor([example.label for example in batch], dtype=torch.float32, device=device)
        targets = torch.tensor([(example.path_score - mean) / std for example in batch],
                               dtype=torch.float32, device=device)
        binary = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(positive_weight, device=device)
        )
        huber = F.smooth_l1_loss(scores, targets)
        target_difference = targets[:, None] - targets[None, :]
        prediction_difference = scores[:, None] - scores[None, :]
        mask = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
        mask &= target_difference.abs() > 1e-6
        ranking = (F.softplus(-target_difference.sign() * prediction_difference)[mask].mean()
                   if mask.any() else scores.sum() * 0.0)
        weights = protocol["loss"]
        loss = (weights["binary_BCE"] * binary + weights["continuous_Huber"] * huber
                + weights["pairwise_rank_softplus"] * ranking)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite boosted severity loss")
        loss.backward()
        norms = {}
        for name, parameter in head.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient: {name}")
                norms[name] = parameter.grad.norm().item()
        norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in head.parameters() if parameter.requires_grad],
            protocol["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        return {"loss": float(loss.detach()), "binary_BCE": float(binary.detach()),
                "continuous_Huber": float(huber.detach()),
                "pairwise_rank": float(ranking.detach()), "gradient_norm": float(norm),
                "parameter_gradient_norms": norms, "targets": len(batch),
                "cached_window_reads": sum(len(example.indices) for example in batch)}
    return step


def make_score_fn(score_mean, score_std):
    mean, std = float(score_mean), float(score_std)

    @torch.no_grad()
    def score(head, tokens, examples, device, micro):
        head.eval()
        rows = []
        for example in examples:
            output = head.forward_outputs(token_inputs(tokens, [example], head.task, device))
            logit = float(output["binary_logit"].item())
            probability = float(torch.sigmoid(output["binary_logit"]).item())
            predicted_score = float(output["continuous_score"].item() * std + mean)
            rows.append({"sample_id": example.sample_id, "subject_id": example.subject_id,
                         "session": example.session, "window_indices": ",".join(map(str, example.indices)),
                         "logit": logit, "score": probability, "predicted_path_score": predicted_score,
                         "path_score": example.path_score, "threshold": 0.5,
                         "y_pred": int(probability >= 0.5)})
        labels = {example.sample_id: example.label for example in examples}
        rows = [{**row, "y_true": labels[row["sample_id"]],
                 "correct": int(row["y_pred"] == labels[row["sample_id"]])} for row in rows]
        result = metrics_from_rows(rows)
        truth = np.asarray([row["path_score"] for row in rows], dtype=np.float64)
        prediction = np.asarray([row["predicted_path_score"] for row in rows], dtype=np.float64)
        truth_rank = np.argsort(np.argsort(truth))
        prediction_rank = np.argsort(np.argsort(prediction))
        result["metrics"]["n_subjects"] = len({example.subject_id for example in examples})
        result["continuous"] = {
            "MAE": float(np.abs(prediction - truth).mean()),
            "RMSE": float(np.sqrt(np.square(prediction - truth).mean())),
            "rank_correlation": float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
            if len(rows) > 1 else None,
        }
        return rows, result
    return score


def make_diagnostics(score_fn):
    @torch.no_grad()
    def diagnostics(head, tokens, examples, device, micro):
        values = token_inputs(tokens, [examples[0]], head.task, device)
        outputs = head.forward_outputs(values)
        result = {
            "diagnostic_sample_windows": len(examples[0].indices),
            "base_logit_rms": float(outputs["base_logit"].square().mean().sqrt()),
            "base_score_rms": float(outputs["base_score"].square().mean().sqrt()),
            "base_parameters_trainable": any(
                parameter.requires_grad for name, parameter in head.named_parameters()
                if name.startswith(head._base_prefixes())
            ),
        }
        if head.config == "base":
            return result
        result.update(
            residual_logit_rms=float(outputs["residual_logit"].square().mean().sqrt()),
            residual_score_rms=float(outputs["residual_score"].square().mean().sqrt()),
        )
        layer = head.mapping[1]
        if not isinstance(layer, NormalizedFractionalDoGKAN):
            return result
        mapped_input = head.mapping[0](values).reshape(-1, 525)[:256]
        bases = layer.basis_features(mapped_input)
        components = layer.component_outputs(mapped_input)
        result.update(
            fractional_order_mean=float(layer.fractional_orders().mean()),
            fractional_order_min=float(layer.fractional_orders().min()),
            fractional_order_max=float(layer.fractional_orders().max()),
            basis_gates=(layer.basis_gate_logits.softmax(0) * layer.basis_count).cpu().tolist(),
            basis_rms=[float(value.square().mean().sqrt()) for value in bases],
            component_output_rms=[float(value.square().mean().sqrt()) for value in components],
        )
        if layer.include_dog:
            result["dog_scale_mixture_mean"] = layer.dog_scale_mix_logits.softmax(-1).mean(0).cpu().tolist()
            state = cpu_state(head)
            layer.basis_weight[:, 2 * layer.input_dim:3 * layer.input_dim].zero_()
            _, result["counterfactual_no_dog"] = score_fn(head, tokens, examples, device, micro)
            head.load_state_dict(state)
        return result
    return diagnostics
