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
    def __init__(self, cond_dim, z_dim, hidden_dim, out_dim, num_layers=1, dropout=0.0):
        super().__init__()
        # init hidden state from condition summary + z
        self.cond_enc = GRUEncoder(cond_dim, hidden_dim, num_layers=1, bidir=False)
        self.init_mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
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

    def forward(self, cond_seq, z, x_future):
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
        # self.cond_enc = GRUEncoder(T_pred, cond_dim)
        # cond_vec = self.cond_enc(cond_seq)
        cond_vec = self.cond_enc(cond_seq)
        #cond_vec = cond_seq.mean(dim=1)  # (B, D_c)   # simple average pooling as condition summary
        h0 = self.init_mlp(torch.cat([cond_vec, z], dim=-1)).unsqueeze(0)  # (1,B,H)
        outputs = []
        x_prev = x_future[:,0,:]  # x0
        outputs = [x_prev.unsqueeze(1)]
        for t in range(T_pred-1):
            cond_t = cond_seq[:, t, :]  # (B, D_c)
            dec_in = torch.cat([cond_t,x_prev], dim=-1).unsqueeze(1)  # (B,1,D_c+D_x)
            dec_out, h0 = self.gru(dec_in, h0)                         # (B,1,H)
            x_pred = self.out(dec_out.squeeze(1))                      # (B,D_x)
            outputs.append(x_pred.unsqueeze(1))
            x_prev = x_pred                                            # feed next step
            # if (torch.rand(1).item() < teacher_force_ratio and t < T_pred - 1):
            #     x_prev = x_future[:, t+1, :]  # 使用真实下一步
            # else:
            #     x_prev = x_pred
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
        self.fut_enc  = GRUEncoder(traj_dim + cond_dim, fut_hidden,  num_layers, bidir=bidir_fut,  dropout=dropout)
        self.to_mu = nn.Linear(fut_hidden, z_dim)
        self.to_logvar = nn.Linear(fut_hidden, z_dim)

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
        enc_in = torch.cat([x_future, cond_seq], dim=-1)  # (B,T,D_x+D_c)
        f_vec = self.fut_enc(enc_in)
        mu, logvar = self.to_mu(f_vec), self.to_logvar(f_vec)
        return mu, logvar

    def decode(self, cond_seq, z, x0):
        return self.decoder(cond_seq, z, x0)

    def forward(self, x_future, cond_seq):
        """
        Training forward pass.
        Args:
            cond_seq: (B, T_pred, D_c)
            x_future: (B, T_pred, D_x)
        """
        mu, logvar = self.encode(x_future, cond_seq)
        z = reparameterize(mu, logvar)

        # use the first frame of ground truth trajectory as starting state
        y_hat = self.decode(cond_seq, z, x_future)
        
        return y_hat, mu, logvar

    @torch.no_grad()
    def sample(self, cond_seq, x0, T_pred=None):
        """Generate trajectories from the prior.

        Args:
            cond_seq (Tensor): Condition sequence of shape ``(B, T, D_c)``.
            x0 (Tensor): Initial pose/state of shape ``(B, D_x)``.
            T_pred (int, optional): Length of the trajectory to generate. If
                ``None`` it defaults to the length of ``cond_seq``.
            n_samples (int): Number of samples to draw per input in the batch.

        Returns:
            Tensor: Samples with shape ``(B, n_samples, T, D_x)``.
        """

        B, T, _ = cond_seq.shape
        if T_pred is None:
            T_pred = T

        device = cond_seq.device
        z_dim = self.to_mu.out_features

        # Repeat condition and initial state for each requested sample.
        cond_seq_rep = cond_seq.unsqueeze(1).expand(B, T, -1)
        cond_seq_rep = cond_seq_rep.reshape(B, T, -1)

        x0_rep = x0.unsqueeze(1).expand(B, -1)
        x0_rep = x0_rep.reshape(B, 1, -1)
        # print('sample:',cond_seq_rep.shape, x0_rep.shape)
        # Sample from the standard normal prior.
        z = torch.randn(B, z_dim, device=device)

        y_hat = self.decode(cond_seq_rep[:, :T_pred, :], z, x0_rep)
        return y_hat.view(B, T_pred, -1)
    
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