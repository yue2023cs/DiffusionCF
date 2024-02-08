# ============================================================================
# Paper:
# Author(s):
# Create Time: 12/10/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

class M3_GEV(nn.Module):
    def __init__(self, n_features, sequence_len, batch_size, n_hidden, n_layers, boundary_tolerance):
        super(M3_GEV, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.sequence_len = sequence_len
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.0)
        self.fcn = nn.Linear(in_features=n_hidden * 2, out_features=4)
        self.fcn2 = nn.Linear(in_features=4, out_features=8)
        self.linear_y = nn.Linear(in_features=8, out_features=1)

        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        self.boundary_tolerance = boundary_tolerance

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device),
            torch.zeros(self.n_layers * 2, self.batch_size, self.n_hidden).to(device)
        )

    def forward(self, input_tensor, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix):
        self.reset_hidden_state()
        self.mu_fix = mu_fix
        self.sigma_fix = sigma_fix
        self.xi_p_fix = xi_p_fix
        self.xi_n_fix = xi_n_fix

        if multiTimeSeries == 'multivariate':
            input_tensor = input_tensor.transpose(1, 2)
        lstm_out, self.hidden = self.lstm(input_tensor, self.hidden)                                                    # lstm_out (batch_size, seq_len, hidden_size*2)
        out = lstm_out[:, -1, :]                                                                                        # getting only the last time step's hidden state of the last layer
        out = self.fcn(out)                                                                                             # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi

        mu = out[:, 0] - self.mu_fix                                                                                    # mu: first node of the fully connected network
        p1 = out[:, 1]                                                                                                  # sigma: second node of the fully connected network
        p2 = out[:, 2]
        p3 = out[:, 3]
        p2 = self.softplus(p2)
        p3 = self.softplus(p3)
        sigma = self.softplus(p1) - self.sigma_fix
        xi_p = ((sigma / (mu - y_min)) * (1 + self.boundary_tolerance) - (p2)) - self.xi_p_fix
        xi_n = ((p3) - (sigma / (y_max - mu)) * (1 + self.boundary_tolerance)) - self.xi_n_fix
        xi_p[xi_p > 0.95] = torch.tensor(0.95)

        out = self.fcn2(out)
        yhat = self.linear_y(out)

        return mu, sigma, xi_p, xi_n, yhat

class ImplementDeepExtrema:
    def __init__(self, batch_size, lr, n_hidden, n_layers, num_epochs):
        self.count_constraint_violation = []

        self.batch_size = batch_size
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_epochs = num_epochs

        self.boundary_tolerance = 0.2

        self.train_history = [0] * self.num_epochs
        self.validation_history = [0] * self.num_epochs
        self.test_history = [0] * self.num_epochs

        if multiTimeSeries == 'multivariate':
            self.n_features = len(F) + 1
        else:
            self.n_features = 1

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, lambda_, lambda_2):
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = self.batch_size, worker_init_fn = self.seed_worker, shuffle=True)
        validation_loader = DataLoader(TensorDataset(X_val, y_val), batch_size = self.batch_size, worker_init_fn = self.seed_worker)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size = self.batch_size, worker_init_fn = self.seed_worker)

        model = M3_GEV(self.n_features, predictorTimesteps * self.n_features, self.batch_size, self.n_hidden, self.n_layers, self.boundary_tolerance)
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        small_value = torch.tensor(0.05)
        zero_tensor = torch.tensor(0.0)
        q1 = torch.tensor(0.05)
        q2 = torch.tensor(0.95)

        y_max = y_train.max()
        y_min = y_train.min()

        mu_hat_all = torch.empty(0).to(device)
        sigma_hat_all = torch.empty(0).to(device)
        xi_hat_all = torch.empty(0).to(device)
        y_all = torch.empty(0).to(device)
        y_hat_all = torch.empty(0).to(device)
        y_q1_all = torch.empty(0).to(device)
        y_q2_all = torch.empty(0).to(device)
        xi_scipy, mu_init, sigma_init = torch.tensor(genextreme.fit(y_train.cpu()))

        best_validation_performance = 9999
        for epoch in (range(self.num_epochs)):
            total_train_loss = 0
            total_validation_loss = 0
            total_test_loss = 0

            train_batch_num = 0
            validation_batch_num = 0
            test_batch_num = 0

            with torch.autograd.set_detect_anomaly(True):
                for i, (inputs, labels) in enumerate(train_loader):
                    if(len(inputs) == self.batch_size):
                        if epoch == 0 and i == 0 and lambda_ > 0.0:
                            with torch.no_grad():
                                # mu_temp, sigma_temp, xi_p_temp, xi_n_temp, yhat_temp = model(inputs, y_max, y_min, zero_tensor, zero_tensor, zero_tensor, zero_tensor)
                                # print(f'initial values: mu (mean): {mu_temp.mean().item()}, sigma (mean): {sigma_temp.mean().item()}, xi_p (mean): {xi_p_temp.mean().item()}, xi_n (mean): {xi_n_temp.mean().item()}')
                                # mu_fix = mu_temp - mu_init
                                # sigma_fix = sigma_temp - sigma_init
                                # mu_temp, sigma_temp, xi_p_temp, xi_n_temp, yhat_temp = model(inputs, y_max, y_min, mu_fix, sigma_fix, zero_tensor, zero_tensor)
                                # xi_p_fix = xi_p_temp - xi_p_init
                                # xi_n_fix = xi_n_temp - xi_n_init
                                mu_fix, sigma_fix, xi_p_fix, xi_n_fix = zero_tensor, zero_tensor, zero_tensor, zero_tensor

                        if lambda_ > 0.0:
                            mu, sigma, xi_p, xi_n, yhat = model(inputs, y_max, y_min, mu_fix, sigma_fix, xi_p_fix,xi_n_fix)
                            # y_med = mu + (sigma/xi_p)*(((math.log(2.0))**(-xi_p)) - 1)
                            # y_q1 = mu + (sigma/xi_p)*(((-math.log(q1))**(-xi_p)) - 1)
                            # y_q2 = mu + (sigma/xi_p)*(((-math.log(q2))**(-xi_p)) - 1)
                            if epoch == 0 and i == 0:
                                print(f'initial values after fixing:  mu (mean): {mu.mean().item()}, sigma (mean): {sigma.mean().item()}, xi_p (mean): {xi_p.mean().item()}, xi_n (mean): {xi_n.mean().item()}')
                                # break
                        else:
                            mu, sigma, xi_p, xi_n, yhat = model(inputs, y_max, y_min, zero_tensor, zero_tensor,zero_tensor,zero_tensor)

                        if lambda_ > 0.0:
                            constraint = 1 + (xi_p / sigma) * (labels - mu)
                            self.count_constraint_violation.append(constraint[constraint < small_value].shape[0])
                            gev_loss = self.calculate_nll(labels.cpu(), mu.cpu(), sigma.cpu(), xi_p.cpu(),is_return=True) / (labels.shape[0])
                            xi_rmse_loss = ((xi_p - xi_n) ** 2).mean().sqrt()
                            evt_loss = lambda_2 * gev_loss + (1 - lambda_2) * xi_rmse_loss

                        rmse_loss = ((labels - yhat) ** 2).mean().sqrt()
                        # print(labels.shape, yhat.shape)

                        if lambda_ == 0.0:
                            train_loss = rmse_loss
                        else:
                            train_loss = lambda_ * evt_loss + (1 - lambda_) * rmse_loss
                        # print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(),4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(),4)} | RMSE(y): {round(rmse_loss.item(),4)} | GEV(NLL): {round(gev_loss.item(),4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(),4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(),4), round(xi_p.mean().item(),4),round(xi_n.mean().item(),4)}')

                        if torch.isinf(train_loss.mean()) or torch.isnan(train_loss.mean()):
                            print("Constraint:\n", constraint, "GEV Loss:\n", gev_loss)
                            print("xi_p \n", xi_p, "ytruth \n", labels, "yhat \n", yhat)
                            # break
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                        total_train_loss += train_loss.item()
                        train_batch_num += 1

                self.train_history[epoch] = total_train_loss/train_batch_num if train_batch_num > 0 else 0.0

                # if lambda_ > 0.0:
                #     print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)} | EVT(NLL+RMSE(xi)): {round(evt_loss.item(), 4)} | RMSE(y): {round(rmse_loss.item(), 4)} | GEV(NLL): {round(gev_loss.item(), 4)} | RMSE(xi_p_n): {round(xi_rmse_loss.item(), 4)}| mu  sigma  xi_p xi_n: {round(mu.mean().item(), 4), round(sigma.mean().item(), 4), round(xi_p.mean().item(), 4), round(xi_n.mean().item(), 4)} | constraint: {constraint.mean().item()}')
                # else:
                #     print(f'Epoch {epoch}  | Loss: | Training: {round(train_loss.item(), 4)}')

                for j, (inputs, labels) in enumerate(validation_loader):
                    if (len(inputs) == self.batch_size):
                        with torch.no_grad():
                            if lambda_ > 0.0:
                                mu, sigma, xi_p, xi_n, y_validation_predict = model(inputs, y_max, y_min, mu_fix, sigma_fix,xi_p_fix, xi_n_fix)
                            else:
                                mu, sigma, xi_p, xi_n, y_validation_predict = model(inputs, y_max, y_min,zero_tensor,zero_tensor, zero_tensor,zero_tensor)
                            rmse_loss = ((y_validation_predict - labels) ** 2).mean().sqrt()
                            validation_loss = rmse_loss

                        total_validation_loss += validation_loss.item()
                        validation_batch_num += 1

                self.validation_history[epoch] = total_validation_loss / validation_batch_num if validation_batch_num > 0 else 0.0

                for k, (inputs, labels) in enumerate(test_loader):
                    if (len(inputs) == self.batch_size):
                        with torch.no_grad():
                            if lambda_ > 0.0:
                                mu, sigma, xi_p, xi_n, y_test_predict = model(inputs, y_max, y_min, mu_fix, sigma_fix, xi_p_fix, xi_n_fix)
                            else:
                                mu, sigma, xi_p, xi_n, y_test_predict = model(inputs, y_max, y_min, zero_tensor, zero_tensor, zero_tensor, zero_tensor)
                            rmse_loss = ((y_test_predict - labels) ** 2).mean().sqrt()
                            test_loss = rmse_loss

                            if (epoch == self.num_epochs - 1):
                                if lambda_ > 0.0:
                                    # y_med = mu + (sigma/xi_p)*(((math.log(2.0))**(-xi_p)) - 1)
                                    y_q1 = mu + (sigma / xi_p) * (((-math.log(q1)) ** (-xi_p)) - 1)
                                    y_q2 = mu + (sigma / xi_p) * (((-math.log(q2)) ** (-xi_p)) - 1)

                                    mu_hat_all = torch.cat((mu_hat_all, mu), 0)
                                    sigma_hat_all = torch.cat((sigma_hat_all, sigma), 0)
                                    xi_hat_all = torch.cat((xi_hat_all, xi_p), 0)
                                    y_q1_all = torch.cat((y_q1_all, y_q1), 0)
                                    y_q2_all = torch.cat((y_q2_all, y_q2), 0)
                                y_all = torch.cat((y_all, labels), 0)
                                y_hat_all = torch.cat((y_hat_all, y_test_predict), 0)

                        total_test_loss += test_loss.item()
                        test_batch_num += 1

                self.test_history[epoch] = total_test_loss / test_batch_num if test_batch_num > 0 else 0.0

                print(self.train_history[epoch], self.validation_history[epoch], self.test_history[epoch])

                if self.validation_history[epoch] <= best_validation_performance:
                    best_validation_performance = self.validation_history[epoch]
                    print("saved", best_validation_performance)
                    torch.save(model, modelSavePath)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def calculate_nll(self, block_maxima, mu, sigma, xi, name="Test", is_return=False):
        size = block_maxima.shape[0]
        block_maxima = torch.flatten(block_maxima.cpu())
        if not torch.is_tensor(mu):
            mu = torch.from_numpy(mu).float().to(device)
        if not torch.is_tensor(sigma):
            sigma = torch.from_numpy(sigma).float().to(device)
        if not torch.is_tensor(xi):
            xi = torch.from_numpy(xi).float().to(device)
        if mu.numel() == 1:
            mu = torch.flatten(torch.full((size, 1), mu))
        if sigma.numel() == 1:
            sigma = torch.flatten(torch.full((size, 1), sigma))
        if xi.numel() == 1:
            xi = torch.full((size, 1), xi)
        mu = torch.flatten(mu).cpu()
        sigma = torch.flatten(sigma).cpu()
        xi = torch.flatten(xi).cpu()

        # using library
        log_pdf = genextreme.logpdf(block_maxima, loc=mu.detach().numpy(), scale=sigma.detach().numpy(), c=-xi.detach().numpy())
        log_likelihood = np.sum(log_pdf)
        # using vector
        # print(xi.shape, block_maxima.shape, mu.shape, sigma.shape)
        constraint = 1 + (xi / sigma) * (block_maxima - mu)
        # constraint = constraint[constraint>0]
        constraint[constraint < 0.05] = torch.tensor(0.5)
        first_term = torch.sum(torch.log(sigma))
        second_term = (torch.sum((1 + 1 / xi) * torch.log(constraint)))
        third_term = torch.sum(constraint ** (-1 / xi))
        nll = (first_term + second_term + third_term)
        if is_return:
            return nll
        else:
            print("\n" + name + ": \n")
            print("negative log likelihood using library:", -log_likelihood, " and using vector:", nll.item())
            print(f"first_term: {first_term}, second_term: {second_term}, third_term: {third_term}")
