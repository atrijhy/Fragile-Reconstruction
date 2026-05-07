import torch
import torch.nn.functional as F


@torch.no_grad()
def _fake_prob_from_logits(logits: torch.Tensor, fake_label: int = 1) -> torch.Tensor:
    if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
        p = logits.view(-1)
        if torch.any(p < 0) or torch.any(p > 1):
            p = torch.sigmoid(p)
        return p.clamp(0.0, 1.0)
    if logits.dim() == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, fake_label].view(-1)
    return torch.softmax(logits, dim=1)[:, fake_label].view(-1)


def apgd_ce_logged(
    model,
    x,
    y,
    eps: float,
    n_iter: int = 10,
    alpha: float | None = None,
    norm: str = 'Linf',
    log_every: int = 1,
    prob_low: float = 0.1,
    prob_high: float = 0.9,
    real_label: int = 0,
    fake_label: int = 1,
    # Early-stop controls
    early_stop: bool = True,
    success_rate_to_stop: float = 1.0,
    real_succ_prob: float = 0.95,   # for y==real_label: success if p_fake > 0.95 (strict)
    fake_succ_prob: float = 0.05,   # for y==fake_label: success if p_fake < 0.05 (strict)
):
    """
    Minimal PGD-CE style attacker with per-iteration logging to approximate APGD-CE behavior.
    - x in [0,1], Linf eps.
    - alpha default = eps / 4 if not set.
    - Prints fake probability after each iteration (every log_every).
    Early stop semantics:
      - Per-sample: once a sample reaches success, we stop updating it in later iterations.
      - Global break: if (num_success / batch_size) >= success_rate_to_stop, we stop the loop early.
    Returns: x_adv tensor.
    """
    assert norm == 'Linf', 'logged attacker currently supports Linf only.'
    device = x.device
    x0 = x.detach()
    x_adv = x0.clone().detach().requires_grad_(True)
    if alpha is None:
        alpha = float(eps) / 4.0
        print(f"[APGD-CE logged] default alpha set to {alpha:.6f} (eps/4)")

    # track success to freeze successful samples
    B = x0.shape[0]
    success_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for i in range(1, n_iter + 1):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)
        with torch.no_grad():
            step = alpha * torch.sign(grad)
            # Do not update successful samples
            if success_mask.any():
                step[success_mask] = 0
            x_adv = x_adv + step
            x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
            x_adv.clamp_(0.0, 1.0)
        x_adv.requires_grad_(True)

        # Evaluate current success and log every iteration (or per log_every)
        with torch.no_grad():
            logits_i = model(x_adv)
            p_fake = _fake_prob_from_logits(logits_i, fake_label=fake_label)

            # update success mask by label-specific strict thresholds
            if p_fake.numel() == B:
                real_mask = (y == real_label)
                fake_mask = (y == fake_label)
                succ_now = torch.zeros_like(success_mask)
                # real samples succeed if p_fake > real_succ_prob (strict)
                if real_mask.any():
                    succ_now[real_mask] = p_fake[real_mask] > real_succ_prob
                # fake samples succeed if p_fake < fake_succ_prob (strict)
                if fake_mask.any():
                    succ_now[fake_mask] = p_fake[fake_mask] < fake_succ_prob
                # other labels: keep as False
                success_mask |= succ_now

            # logging
            if (i % max(1, log_every)) == 0 or i == 1:
                if p_fake.numel() == 1:
                    print(
                        f"[APGD-CE logged] iter {i}/{n_iter}: p_fake {p_fake.item():.6f}  (<= {prob_low}: {(p_fake.item()<=prob_low)}, >= {prob_high}: {(p_fake.item()>=prob_high)}), "
                        f"success_rate {success_mask.float().mean().item():.3f}"
                    )
                else:
                    mean_p = p_fake.mean().item()
                    below = int((p_fake <= prob_low).sum().item())
                    above = int((p_fake >= prob_high).sum().item())
                    succ = success_mask.sum().item()
                    print(
                        f"[APGD-CE logged] iter {i}/{n_iter}: mean(p_fake) {mean_p:.6f}  count<=low {below}  count>=high {above}  "
                        f"success {succ}/{B} ({succ/float(B):.2%})"
                    )

            # global early stop
            if early_stop and success_mask.float().mean().item() >= success_rate_to_stop:
                print(f"[APGD-CE logged] early stop at iter {i}: success_rate reached {success_mask.float().mean().item():.3f} >= {success_rate_to_stop}")
                break

    return x_adv.detach()
