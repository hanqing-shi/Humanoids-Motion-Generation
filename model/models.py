import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUEncoder(nn.Module):
    """
    Encodes a sequence (B, T, D) into a fixed hidden vector (B, H).
    Standard GRU implementation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, input_dim)
        # h: (num_layers, B, hidden_dim)
        _, h = self.gru(x)
        
        # We assume 1 layer for simplicity as per original code. 
        # If num_layers > 1, we typically take the last layer's state.
        return h[-1] # (B, hidden_dim)


class GRUDecoder(nn.Module):
    """
    Deterministic GRU Decoder.
    Predicts trajectory using the Encoder's hidden state as initialization.
    """
    def __init__(self, 
                 cond_dim: int, 
                 hidden_dim: int, 
                 out_dim: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0,
                 scale_pos_z: float = 0.1,
                 scale_joints: float = 0.2):
        super().__init__()
        
        # Configurable scales to avoid magic numbers in forward loop
        self.scale_pos_z = scale_pos_z
        self.scale_joints = scale_joints

        # Project encoder hidden state to match decoder hidden state size (if needed)
        # Here we assume dimensions match or use a simple linear layer
        self.init_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU() # Modern activation

        # GRU Input = Condition (Current) + State (Previous)
        self.gru = nn.GRU(
            input_size=cond_dim + out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond_seq, h_enc, x0, x_future=None, teacher_forcing_ratio=0.0):
        """
        Args:
            cond_seq: (B, T_pred, cond_dim) Future conditions (joystick)
            h_enc: (B, hidden_dim) Hidden state from Encoder
            x0: (B, out_dim) Initial state (last frame of past)
            x_future: (B, T_pred, out_dim) Ground truth for teacher forcing
            teacher_forcing_ratio: float
        """
        B, T_pred, _ = cond_seq.shape
        
        # Initialize hidden state: (B, H) -> (1, B, H)
        h0 = self.init_proj(h_enc)
        h0 = self.act(h0).unsqueeze(0) 

        outputs = []
        x_prev = x0
        
        for t in range(T_pred):
            cond_t = cond_seq[:, t, :]  # (B, D_c)
            
            # Input: Concatenate condition and previous state
            dec_in = torch.cat([cond_t, x_prev], dim=-1).unsqueeze(1)  # (B, 1, D_c+D_x)
            
            # GRU Step
            dec_out, h0 = self.gru(dec_in, h0)
            
            # Raw projection
            raw = self.out_proj(dec_out.squeeze(1)) # (B, D_x)
            
            # --- Hybrid Prediction Strategy ---
            # 1. Velocity (vel_x, vel_y): Direct prediction [-1, 1]
            next_vel = torch.tanh(raw[:, 0:2])
            
            # 2. Root Z: Residual update
            delta_z = self.scale_pos_z * torch.tanh(raw[:, 2:3])
            next_z = x_prev[:, 2:3] + delta_z
            
            # 3. Quaternion: Direct prediction + Normalize
            # Indices 3:7 correspond to [qx, qy, qz, qw]
            next_quat = F.normalize(raw[:, 3:7], dim=-1)
            
            # 4. Joints: Residual update
            delta_joints = self.scale_joints * torch.tanh(raw[:, 7:])
            next_joints = x_prev[:, 7:] + delta_joints
            
            # Reassemble
            x_pred = torch.cat([next_vel, next_z, next_quat, next_joints], dim=-1)
            outputs.append(x_pred.unsqueeze(1))

            # Teacher Forcing Logic
            use_gt = (x_future is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_gt and (t < T_pred - 1):
                x_prev = x_future[:, t+1, :]
            else:
                x_prev = x_pred

        return torch.cat(outputs, dim=1)


class TrajCVAE(nn.Module):
    """
    Sequence-to-Sequence Trajectory Predictor.
    (Name kept as TrajCVAE for compatibility, but strictly it is now a deterministic Seq2Seq)
    Structure:
        Encoder: [Past State, Past Cond] -> Hidden State
        Decoder: Hidden State + [Future Cond] -> Future State
    """
    def __init__(
        self,
        traj_dim: int,
        cond_dim: int,
        teacher_forcing: float = 0.0,
        past_lenth: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.past_lenth = past_lenth
        self.teacher_forcing = teacher_forcing
        
        # Encoder
        self.encoder = GRUEncoder(
            input_dim=traj_dim + cond_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout
        )
        
        # Decoder
        self.decoder = GRUDecoder(
            cond_dim=cond_dim, 
            hidden_dim=hidden_dim, 
            out_dim=traj_dim,
            num_layers=num_layers, 
            dropout=dropout
        )

    def forward(self, x, cond_seq, teacher_forcing_ratio=None):
        """
        Training Forward Pass.
        Returns:
            y_hat: (B, T_total, D) The full reconstructed trajectory (Past + Future)
        """
        # Split Data into Past and Future
        x_past = x[:, :self.past_lenth, :]
        x_future = x[:, self.past_lenth:, :]
        
        cond_past = cond_seq[:, :self.past_lenth, :]
        cond_future = cond_seq[:, self.past_lenth:, :]
        
        # 1. Encode Past -> Hidden State
        enc_in = torch.cat([x_past, cond_past], dim=-1)
        h_enc = self.encoder(enc_in) # (B, H)
        
        # 2. Decode Future
        x0 = x_past[:, -1, :] # Initial state for decoder is the last past frame
        
        tf = teacher_forcing_ratio if teacher_forcing_ratio is not None else self.teacher_forcing
        
        # Note: z is removed completely
        y_future = self.decoder(cond_future, h_enc, x0, x_future=x_future, teacher_forcing_ratio=tf)
        
        # 3. Concatenate Past (Ground Truth) + Future (Predicted)
        # Returning full sequence helps with visualization alignment
        y_full = torch.cat([x_past, y_future], dim=1)
        
        # Return format: pred, mu(None), logvar(None) to keep interface consistent if needed
        return y_full, None, None

    @torch.no_grad()
    def sample(self, cond_past, cond_future, x_past, x_future=None):
        """
        Inference Forward Pass.
        """
        B, T, _ = cond_future.shape
        
        # 1. Encode
        enc_in = torch.cat([x_past, cond_past], dim=-1)
        h_enc = self.encoder(enc_in)
        
        # 2. Decode
        x0 = x_past[:, -1, :]
        y_future = self.decoder(cond_future, h_enc, x0, x_future=None, teacher_forcing_ratio=0.0)

        return y_future
       
class AMDM(nn.Module):
    """
    Auto-Regressive Motion Diffusion Model (A-MDM).
    Generates current frame x_curr conditioned on previous frame x_prev and condition c.
    """
    def __init__(self, state_dim, cond_dim, hidden_dim=1024, T=40):
        super().__init__()
        self.state_dim = state_dim
        self.T = T
        
        # Time embedding to inform the model of the current diffusion step
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Lightweight MLP architecture
        # Input: [Noisy current frame, Previous frame, Time embedding, Task condition]
        self.decoder = nn.Sequential(
            nn.Linear(state_dim * 2 + 128 + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim) 
        )

        # DDPM Variance Schedule [cite: 1223]
        betas = torch.linspace(0.0001, 0.02, T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Pre-compute constants for diffusion [cite: 1221, 1224]
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("reciprocal_sqrt_alphas", torch.sqrt(1. / alphas))
        self.register_buffer("remove_noise_coeff", betas / torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("sigma", torch.sqrt(betas))

    def forward(self, x_prev, x_curr, c):
        """ Training: Returns MSE loss between predicted and ground-truth noise [cite: 1232] """
        t = torch.randint(0, self.T, (x_curr.shape[0], 1), device=x_curr.device)
        noise = torch.randn_like(x_curr)
        
        # Forward Diffusion Process [cite: 1221, 1224]
        xt = (self.sqrt_alphas_cumprod[t] * x_curr + 
              self.sqrt_one_minus_alphas_cumprod[t] * noise)
        
        time_emb = self.time_mlp(t.float() / self.T)
        pred_noise = self.decoder(torch.cat([xt, x_prev, time_emb, c], dim=-1))
        
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_step(self, x_prev, c):
        """ Inference: Generates x_curr from x_prev via T denoising steps [cite: 1233, 1235] """
        x_t = torch.randn_like(x_prev) 
        for t_idx in reversed(range(self.T)):
            t = torch.full((x_prev.shape[0], 1), t_idx, device=x_prev.device)
            time_emb = self.time_mlp(t.float() / self.T)
            pred_noise = self.decoder(torch.cat([x_t, x_prev, time_emb, c], dim=-1))
            
            # Reverse Denoising Step [cite: 1233, 1234]
            mean = (self.reciprocal_sqrt_alphas[t_idx] * (x_t - self.remove_noise_coeff[t_idx] * pred_noise))
            if t_idx > 0:
                x_t = mean + self.sigma[t_idx] * torch.randn_like(x_t)
            else:
                x_t = mean
        return x_t