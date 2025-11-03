# models.py
import torch
import torch.nn as nn

class TrajCVAE(nn.Module):
    """
    GRU-based CVAE that reconstructs core motion state (root + joints),
    optionally conditioned on root v/ω, with an optional auxiliary head
    for body_pos_local supervision.
    """

    def __init__(self, target_dim, cond_dim=0, hidden=256, latent=64,
                 num_layers=2, dropout=0.1, aux_dim: int = 0):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.latent = latent
        self.aux_dim = aux_dim

        enc_in = target_dim + cond_dim
        self.encoder = nn.GRU(enc_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.to_mu = nn.Linear(hidden, latent)
        self.to_logv = nn.Linear(hidden, latent)

        dec_in = target_dim + cond_dim + latent
        self.decoder = nn.GRU(dec_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)

        self.out_core = nn.Linear(hidden, target_dim)
        self.out_aux  = nn.Linear(hidden, aux_dim) if aux_dim > 0 else None

    def encode(self, target, cond):
        x = torch.cat([target, cond], dim=-1) if cond is not None else target
        h, _ = self.encoder(x)
        h_last = h[:, -1, :]
        mu = self.to_mu(h_last)
        logv = self.to_logv(h_last)
        return mu, logv

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, target, cond, z):
        B, T, D = target.shape
        start = torch.zeros((B, 1, D), dtype=target.dtype, device=target.device)
        tf_in = torch.cat([start, target[:, :-1, :]], dim=1)
        z_rep = z.unsqueeze(1).expand(B, T, z.shape[-1])
        x = [tf_in, z_rep] if cond is None else [tf_in, cond, z_rep]
        dec_in = torch.cat(x, dim=-1)
        h, _ = self.decoder(dec_in)
        y_core = self.out_core(h)
        y_aux  = self.out_aux(h) if self.out_aux is not None else None
        return y_core, y_aux

    def forward(self, target, cond):
        mu, logv = self.encode(target, cond)
        z = self.reparameterize(mu, logv)
        y_core, y_aux = self.decode(target, cond, z)
        return y_core, y_aux, mu, logv
