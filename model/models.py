import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """z = mu + sigma * eps"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

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
    def __init__(self, cond_dim, z_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0, teacher_forcing_ratio=0):
        super().__init__()
        # init hidden state from condition summary + z
        self.cond_enc = GRUEncoder(cond_dim, hidden_dim, num_layers=1, bidir=False)
        self.init_mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.teacher_forcing_ratio = teacher_forcing_ratio
        # GRU input = [current_cond_t, prev_state]
        self.gru = nn.GRU(
            input_size=cond_dim + out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_seq, z, x0, x_future, h_enc):
        B, T_pred, _ = cond_seq.shape
        h0 = self.init_mlp(torch.cat([h_enc, z], dim=-1)).unsqueeze(0)
        outputs = []
        x_prev = x0
        for t in range(T_pred):
            cond_t = cond_seq[:, t, :]  # (B, D_c)
            dec_in = torch.cat([cond_t,x_prev], dim=-1).unsqueeze(1)  # (B,1,D_c+D_x)
            dec_out, h0 = self.gru(dec_in, h0)                         # (B,1,H)

            # x_pred = self.out(dec_out.squeeze(1))                      # (B,D_x)
            # outputs.append(x_pred.unsqueeze(1))
            raw = self.out(dec_out.squeeze(1))      # (B, D_x)
            delta = 0.1 * torch.tanh(raw)           # [-0.1, 0.1]
            x_pred = x_prev + delta
            outputs.append(x_pred.unsqueeze(1))

            if (torch.rand(1).item() < self.teacher_forcing_ratio) and (t < T_pred - 1):
                x_prev = x_future[:, t+1, :]  # teacher forcing
            else:
                x_prev = x_pred
        y_hat = torch.cat(outputs, dim=1)  # (B,T_pred,D_x)
        return y_hat

class MLPEncoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        condition_size,
        feature_size,
    ):
        super().__init__()
        # Encoder
        # Takes (future pose) | condition: (current pose + future velocity label) as input
        input_size = frame_size * feature_size + condition_size
        traj_size = frame_size * feature_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(traj_size + hidden_size, hidden_size)
        self.mu = nn.Linear(traj_size + hidden_size, latent_size)
        self.logvar = nn.Linear(traj_size + hidden_size, latent_size)

    def encode(self, x, c):
        x = x.reshape(x.shape[0], -1)  # flatten
        c = c.reshape(c.shape[0], -1)
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
class MLPDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        condition_size,
        feature_size,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + condition_size
        output_size = feature_size * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)
        self.frame_size = frame_size
        self.feature_size = feature_size

    def decode(self, z, c):
        c = c.reshape(c.shape[0], -1) # flatten
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        out = self.out(torch.cat((z, h5), dim=1))
        return out.reshape(out.shape[0], self.frame_size, self.feature_size)  # (B, T, D)

    def forward(self, z, c):
        return self.decode(z, c)
    
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
        teacher_forcing,
        past_lenth,
        fut_hidden=256,
        z_dim=64,
        dec_hidden=256,
        num_layers=1,
        dropout=0.0,
        bidir_cond=False,
        bidir_fut=False
    ):
        super().__init__()
        # Encoder
        self.past_enc  = GRUEncoder(traj_dim + cond_dim, fut_hidden,  num_layers, bidir=bidir_fut, dropout=dropout)
        self.to_mu = nn.Linear(fut_hidden, z_dim)
        self.to_logvar = nn.Linear(fut_hidden, z_dim)
        self.past_lenth = past_lenth
        # Decoder
        self.decoder = GRUDecoder(
            cond_dim=cond_dim, z_dim=z_dim,
            hidden_dim=dec_hidden, out_dim=traj_dim,
            num_layers=num_layers, dropout=dropout,
            teacher_forcing_ratio = teacher_forcing
        )

    def encode(self, x, cond_seq):
        enc_in = torch.cat([x, cond_seq], dim=-1)  # (B,T,D_x+D_c)
        h_enc = self.past_enc(enc_in)
        mu, logvar = self.to_mu(h_enc), self.to_logvar(h_enc)
        return mu, logvar, h_enc

    def decode(self, cond_seq, z, x0, x_future, h_enc):
        return self.decoder(cond_seq, z, x0, x_future, h_enc)

    def forward(self, x, cond_seq):
        x_past = x[:, :self.past_lenth, :]
        x_future = x[:, self.past_lenth:, :]
        cond_past = cond_seq[:, :self.past_lenth, :]
        cond_future = cond_seq[:, self.past_lenth:, :]
        x0 = x_past[:, -1, :]
        mu, logvar, h_enc = self.encode(x_past, cond_past)
        z = reparameterize(mu, logvar)
        y_hat = self.decode(cond_future, z, x0, x_future, h_enc)
        y_hat = torch.cat([x_past, y_hat], dim=1) # full trajectory
        return y_hat, mu, logvar

    @torch.no_grad()
    def sample(self, cond_past, cond_future, x_past, x_future):
        B, T, _ = cond_future.shape
        device = cond_future.device
        z_dim = self.to_mu.out_features
        x0 = x_past[:, -1, :]
        mu, logvar, h_enc = self.encode(x_past, cond_past)
        z = reparameterize(mu, logvar)

        y_hat = self.decode(cond_future, z, x0, x_future, h_enc)

        return y_hat.view(B, T, -1)
    
class MlpCVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        condition_size,
        feature_size,
        hidden_size=128,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.condition_size = condition_size
        self.feature_size = feature_size

        args = (
            frame_size,
            latent_size,
            hidden_size,
            condition_size,
            feature_size,
        )

        self.encoder = MLPEncoder(*args)
        self.decoder = MLPDecoder(*args)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    def encode(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return mu, logvar
    
    @torch.no_grad()
    def sample(self, z, c):
        return self.decoder(z, c)