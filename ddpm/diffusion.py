from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

class DDPM:
    def __init__(self, cfg: DiffusionConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample_v(self, x0, t, noise):
        """
        Возвращает xt и v_target одновременно
        """
        xt = self.q_sample(x0, t, noise)

        s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
        v_target = s1 * noise - s2 * x0
        return xt, v_target

    def get_v(self, x_0, noise, t):
        """
        Вычисляет целевое значение v по формуле:
        v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x_0

    def predict_x0_from_v(self, x_t, v, t):
        """
        Восстанавливает x_0 из предсказанного v:
        x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        return sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v

    def predict_eps_from_v(self, x_t, v, t):
        """
        Восстанавливает шум (epsilon) из предсказанного v:
        eps = sqrt(1 - alpha_bar) * x_t + sqrt(alpha_bar) * v
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]

        return sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * v

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x_t = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*noise
        """
        b = x0.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        One reverse step: x_{t-1} from x_t
        """
        b = x.shape[0]
        tt = torch.full((b,), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)

        beta_t = self.betas[tt].view(b, 1, 1, 1)
        alpha_t = self.alphas[tt].view(b, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[tt].view(b, 1, 1, 1)

        # mean of posterior
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)
        mean = mean.clamp(-1, 1)

        if t == 0:
            return mean

        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise

    @torch.no_grad()
    def sample(self, model, shape, steps: int | None = None) -> torch.Tensor:
        """
        Generates x0 from pure noise. shape = (B, C, H, W)
        """
        T = self.cfg.timesteps if steps is None else steps
        T = min(T, self.cfg.timesteps)  # <= фикс

        x = torch.randn(shape, device=self.device)

        for t in reversed(range(T)):
            x = self.p_sample(model, x, t)

        return x
