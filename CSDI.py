# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/28/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)

    return layer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0,
        local_attn_window_size = 0,
    )

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=64, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim / 2), persistent=False,)
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = torch.nn.functional.silu(x)
        x = self.projection2(x)
        x = torch.nn.functional.silu(x)

        return x

    def _build_embedding(self, num_steps, dim=32):
        steps = torch.arange(num_steps).unsqueeze(1)                                                                    # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)                                        # (1,dim). In order to encode the position information of elements in sequence, used for Transformer
        table = steps * frequencies                                                                                     # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)                                           # (T,dim*2)

        return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim = 2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(num_steps = config["num_steps"], embedding_dim = config["diffusion_embedding_dim"],)

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):                                                                    # cond_info: side_info + cond_mask [16,128+1,24]
        B, inputdim, L = x.shape

        x = self.input_projection(x)
        x = torch.nn.functional.relu(x)
        x = x.reshape(B, self.channels, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, L)
        x = self.output_projection1(x)                                                                                  # (B,channel,L)
        x = torch.nn.functional.relu(x)
        x = self.output_projection2(x)                                                                                  # (B,1,L)
        x = x.reshape(B, L)

        return x

    def get_torch_trans(heads=8, layers=1, channels=64):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
        )

        return nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def get_linear_trans(heads=8, layers=1, channels=64, localheads=0, localwindow=0):
        return LinearAttentionTransformer(
            dim=channels,
            depth=layers,
            heads=heads,
            max_seq_len=256,
            n_local_attn_heads=0,
            local_attn_window_size=0,
        )

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, L).permute(0, 1, 2).reshape(B, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, channel, L).permute(0, 1, 2).reshape(B, channel, L)

        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip


class CSDI(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim).to(device)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1)

    def time_embedding(self, observed_tp, d_model=128):
        pe = torch.zeros(observed_tp.shape[0], observed_tp.shape[1], d_model).to(device)
        position = observed_tp.unsqueeze(2).to(device)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model).to(device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        side_info = time_embed.permute(0, 2, 1)  # (B,*,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1).to(device)  # (B,1,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, L = observed_data.shape

        if is_train != 1:                                                                                               # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]                                                                             # (B,1)
        noise = torch.randn_like(observed_data)                                                                         # use standard Gaussian to generate a noise tensor of the same shape
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)                                                           # (B,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)                                                             # For two matrices, it is element-wise multiplication
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)                                                 # (B,2,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input.to(device), side_info.to(device), torch.tensor([t]).to(device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, observed_tp, observed_data, observed_mask, gt_mask, is_train=1):
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp.to(device), cond_mask.to(device))
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, it, n_samples = 1):
        for batch_no, (observed_tp, observed_data, observed_mask, gt_mask) in enumerate(it, start=1):
            with torch.no_grad():
                cond_mask = gt_mask
                target_mask = observed_mask - cond_mask
                side_info = self.get_side_info(observed_tp, cond_mask)
                samples = self.impute(observed_data.to(device), cond_mask.to(device), side_info.to(device), n_samples)

            return samples, observed_data, target_mask, observed_mask, observed_tp
