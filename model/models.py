import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """z = mu + sigma * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu, logvar):
    """KL( q(z|.) || N(0,I) ) per-sample"""
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1)


class GRUEncoder(nn.Module):
    """Encodes a sequence (B,T,D) -> fixed vector (B,H)"""
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidir=False, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidir,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.bidir = bidir
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out, h = self.gru(x)
        if self.bidir:
            h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)
        else:
            h_last = h[-1]                              # (B, H)
        return h_last


class GRUDecoder(nn.Module):
    """
    GRU decoder that predicts trajectory given:
      - condition sequence (B, T, D_c)
      - latent z (B, z_dim)
      - initial state x0 (B, D_x)
    """
    def __init__(self, cond_dim, z_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0):
        super().__init__()
        # init hidden state from condition summary + z
        self.init_mlp = nn.Sequential(
            nn.Linear(cond_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # GRU input = [current_cond_t, prev_state]
        self.gru = nn.GRU(
            input_size=cond_dim + out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_seq, z, x0):
        """
        Args:
            cond_seq: (B, T_pred, D_c)  - velocity or other condition sequence
            z:        (B, z_dim)
            x0:       (B, D_x)          - initial state (current state)
        Returns:
            y_hat: (B, T_pred, D_x)
        """
        B, T_pred, _ = cond_seq.shape
        # compute global context for init hidden
        # TODO: how to encode condition
        self.cond_enc = GRUEncoder(T_pred, cond_dim)
        cond_vec = self.cond_enc(cond_seq)

        h0 = self.init_mlp(torch.cat([z, cond_vec], dim=-1)).unsqueeze(0)

        outputs = []
        x_prev = x0  # start from current state

        for t in range(T_pred):
            cond_t = cond_seq[:, t, :]  # (B, D_c)
            dec_in = torch.cat([cond_t, x_prev], dim=-1).unsqueeze(1)  # (B,1,D_c+D_x)
            dec_out, h0 = self.gru(dec_in, h0)                         # (B,1,H)
            x_pred = self.out(dec_out.squeeze(1))                      # (B,D_x)
            outputs.append(x_pred.unsqueeze(1))
            x_prev = x_pred                                            # feed next step

        y_hat = torch.cat(outputs, dim=1)  # (B,T_pred,D_x)
        return y_hat


class TrajCVAE(nn.Module):
    """
    Conditional VAE for trajectory generation.
    Condition = velocity sequence (B,T_cond,D_c)
    Target    = trajectory sequence (B,T_pred,D_x)
    """
    def __init__(
        self,
        traj_dim,          # per-frame feature dimension D_x
        cond_dim,          # per-frame condition dimension D_c
        cond_hidden=256,
        fut_hidden=256,
        z_dim=64,
        dec_hidden=256,
        num_layers=1,
        dropout=0.0,
        bidir_cond=False,
        bidir_fut=False
    ):
        super().__init__()
        # Encoders
        # TODO: we don't need to use GRU to encode condition
        self.cond_enc = GRUEncoder(cond_dim, cond_hidden, num_layers, bidir=bidir_cond, dropout=dropout)
        self.fut_enc  = GRUEncoder(traj_dim, fut_hidden,  num_layers, bidir=bidir_fut,  dropout=dropout)

        cond_vec_dim = cond_hidden * (2 if bidir_cond else 1)
        fut_vec_dim  = fut_hidden  * (2 if bidir_fut  else 1)
        enc_cat_dim  = cond_vec_dim + fut_vec_dim

        self.to_mu     = nn.Linear(enc_cat_dim, z_dim)
        self.to_logvar = nn.Linear(enc_cat_dim, z_dim)

        # Decoder now takes full condition sequence
        self.decoder = GRUDecoder(
            cond_dim=cond_dim, z_dim=z_dim,
            hidden_dim=dec_hidden, out_dim=traj_dim,
            num_layers=num_layers, dropout=dropout
        )

    def encode(self, x_future, cond_seq):
        """
        x_future: (B, T_pred, D_x)
        cond_seq: (B, T_cond, D_c)
        """
        c_vec = self.cond_enc(cond_seq)
        f_vec = self.fut_enc(x_future)
        h = torch.cat([c_vec, f_vec], dim=-1)
        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def decode(self, cond_seq, z, x0):
        return self.decoder(cond_seq, z, x0)

    def forward(self, cond_seq, x_future, beta=1.0):
        """
        Training forward pass.
        Args:
            cond_seq: (B, T_pred, D_c)
            x_future: (B, T_pred, D_x)
        """
        mu, logvar = self.encode(x_future, cond_seq)
        z = reparameterize(mu, logvar)

        # use the first frame of ground truth trajectory as starting state
        x0 = x_future[:, 0, :]

        y_hat = self.decode(cond_seq, z, x0)

        recon = F.mse_loss(y_hat, x_future, reduction="none").mean(dim=(1, 2))
        kl = kl_divergence(mu, logvar)
        loss = (recon + beta * kl).mean()
        return y_hat, loss, {"recon": recon.mean().item(), "kl": kl.mean().item()}

    @torch.no_grad()
    def sample(self, cond_seq, x0, T_pred, n_samples=1):
        """
        Inference: generate trajectory given condition and initial state.
        """
        B = cond_seq.size(0)
        cond_seq = cond_seq.repeat_interleave(n_samples, dim=0)
        z = torch.randn(cond_seq.size(0), self.to_mu.out_features, device=cond_seq.device)
        x0 = x0.repeat_interleave(n_samples, dim=0)
        y_hat = self.decode(cond_seq, z, x0)
        return y_hat.view(B, n_samples, T_pred, -1)

class PoseCVAE(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_pose: int,
        d_condition: int,
        latent_size: int,
        feature_size: int,

    ):
        super.__init__()
        
        self.seq_len = seq_len
        self.d_pose = d_pose
        self.d_condition = d_condition
        self.latent_size = latent_size
        self.feature_size = feature_size

        # encode
        h1 = 256
        # takes pose + condition as input
        self.fc1 = nn.Linear(seq_len*(d_pose + d_condition),h1)
        self.fc21 = nn.Linear(h1, latent_size)
        self.fc22 = nn.Linear(h1, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + d_condition, h1)
        self.fc4 = nn.Linear(h1, feature_size)

    def encode(self,pose_seq, cond_seq):
        """
        Args:
            pose_seq: (B, T, D_pose)
            cond_seq: (B, T, D_condition)
        Returns:
            mu, logvar
        """
        B, T, Dp = pose_seq.shape

        x = torch.cat([pose_seq, cond_seq], dim=-1).reshape(B, T*(Dp + self.d_condition))
        h1 = F.elu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar
    
    def decode(self, z, cond_seq):
        """
        Args:
            z: (B, latent_size)
            cond_seq: (B, T, D_condition)
        Returns:
            x_hat: (B, T, feature_size)
        """
        B, T, Dc = cond_seq.shape
        # pad z to every frame
        z_expand = z.unsqueeze(1).repeat(1,T,1) # (B, T, latent_size)
        inputs = torch.cat([z_expand, cond_seq], dim=-1) # (B, T, latent_size + D_condition)

        h = F.elu(self.fc3(inputs))
        out = self.fc4(h) # (B, T, feature_size)
        return out
    
    def forward(self, pose_seq, cond_seq, beta=1.0):
        """
        Args:
            pose_seq: (B, T, D_pose)
            cond_seq: (B, T, D_condition)
        Returns:
            x_hat: (B, T, feature_size)
            mu, logvar
        """
        mu, logvar = self.encode(pose_seq, cond_seq)
        z = reparameterize(mu, logvar)
        x_hat = self.decode(z, cond_seq)

        # loss
        recon_loss = F.mse_loss(x_hat, pose_seq, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        return x_hat, loss, {"recon": recon_loss.item(), "kl": kl_loss.item()}