import torch


@torch.jit.script
def solve_rc_ode(I, R, C, dt):
    """Solve the RC ODE using the analytical solution."""
    tau = R * C
    decay = torch.exp(-dt / tau)
    gain = R * (1.0 - decay)
    V = torch.zeros_like(I)
    v_curr = torch.zeros(I.shape[0], 1, device=I.device)
    for t in range(I.shape[1]):
        v_next = v_curr * decay + I[:, t : t + 1] * gain
        V[:, t] = v_next.squeeze(1)
        v_curr = v_next
    return V
