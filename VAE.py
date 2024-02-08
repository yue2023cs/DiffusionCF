# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/31/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        # self.x2hidden = nn.LSTM(input_dim, hidden_dim, batch_first=True)                                                # If use LSTM

        self.x2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden2logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden2x = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x = x.unsqueeze(2)                                                                                              # If use LSTM, considering batches here, the last dimension must be 1!!!
        # _, (hidden, _) = self.x2hidden(x)                                                                               # If use LSTM

        hidden = self.x2hidden(x)
        hidden = torch.nn.functional.relu(hidden)
        z_mean = self.hidden2mean(hidden)
        z_logvar = self.hidden2logvar(hidden)

        z_mean = torch.nn.functional.softplus(z_mean)                                                                   # it said that usually the mean does not need activation function?
        z_logvar = torch.nn.functional.softplus(z_logvar)                                                               # Compared with ReLU, the activation function 'softplus' can better avoid zero gradients

        return z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z

    def decode(self, z):
        hidden = self.latent2hidden(z)
        hidden = torch.nn.functional.relu(hidden)
        decoded = self.hidden2x(hidden)

        return decoded

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)

        return self.decode(z), mean, logvar

def sliding_window_transform(matrix, window_size):
    num_rows, num_cols = matrix.shape
    if window_size > num_cols:
        raise ValueError("Window size must be less than or equal to the number of columns in the matrix.")

    subsequences = []
    for row in matrix:
        for i in range(num_cols - window_size + 1):
            subsequences.append(row[i:i+window_size])

    transformed_matrix = np.vstack(subsequences)

    return transformed_matrix

def trainVAE(cfWindowSize, foldername, train_dataset, val_dataset, epochs = 100, batch_size = 32, learning_rate = 0.01):
    input_dim = cfWindowSize
    hidden_dim = 64
    latent_dim = 2                                                                                                      # MARK: people usually use 2 as the latent dimension.
    best_val_loss = float('inf')

    train_dataset = np.squeeze(np.array(train_dataset.cpu()), axis=2)
    train_dataset = sliding_window_transform(train_dataset, cfWindowSize)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = np.squeeze(np.array(val_dataset.cpu()), axis = 2)
    val_dataset = sliding_window_transform(val_dataset, cfWindowSize)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        epochNum = 0

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            reconstructed_batch, mean, logvar = model(data)

            recon_loss = loss_function(reconstructed_batch, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + lambdaVAE * kl_loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            epochNum += 1

        print(data[0], reconstructed_batch[0])

        train_loss = train_loss / epochNum
        print(f'Epoch: {epoch}, Train loss: {train_loss}')

        model.eval()
        val_loss = 0
        epochNum = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                reconstructed_batch, mean, logvar = model(data)
                recon_loss = loss_function(reconstructed_batch, data)
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + lambdaVAE * kl_loss
                val_loss += recon_loss.item()
                epochNum += 1

        val_loss = val_loss / epochNum
        print(f'Epoch: {epoch}, Validation loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, foldername + multiTimeSeries + '_' + str(vae_batch_size) + '_' +str(cfWindowSize) + '_' + X + '_vae_model.pth')
            print(f'Model saved at epoch {epoch} with validation loss: {best_val_loss}')
