import torch
import torch.nn as nn

# No importes nada de aquí mismo

class TB_ODEFunc(nn.Module):
    def __init__(self, latent_dim, static_dim, control_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = latent_dim + static_dim + control_dim
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers-1 else latent_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers-1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, t, x, static, u_func):
        u = u_func(t)
        return self.net(torch.cat([x, static, u], dim=1))

def integrate_rk4(func, x0, t0, t1, dt):
    steps = int((t1 - t0) / dt)
    t = torch.tensor(float(t0), device=x0.device, dtype=x0.dtype)
    x = x0
    for _ in range(steps):
        k1 = func(t, x)
        k2 = func(t + 0.5*dt, x + 0.5*dt*k1)
        k3 = func(t + 0.5*dt, x + 0.5*dt*k2)
        k4 = func(t + dt, x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
    return x

class TB_NeuralODE(nn.Module):
    def __init__(self, latent_dim=256, static_dim=14, control_dim=4):
        super().__init__()
        self.static_dim = static_dim
        self.encoder = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.odefunc = TB_ODEFunc(latent_dim, static_dim, control_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + static_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, static, treatment_seq, dt=0.1, method='dopri5'):
        B, T, _ = treatment_seq.shape
        x0 = self.encoder(static)

        def u_func(t):
            day = torch.floor(t).long().clamp(0, T-1)
            return treatment_seq[:, day, :]

        def func(t, x):
            return self.odefunc(t, x, static, u_func)

        if method == 'dopri5':
            from torchdiffeq import odeint
            t_span = torch.linspace(0.0, float(T), steps=100, device=static.device)
            x_traj = odeint(func, x0, t_span, method='dopri5', rtol=1e-5, atol=1e-6)
            xT = x_traj[-1]
        else:
            xT = integrate_rk4(func, x0, 0.0, float(T), dt)

        out = self.decoder(torch.cat([xT, static], dim=1))
        return out[:, 0], out[:, 1]