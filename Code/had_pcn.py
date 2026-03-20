import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model Components
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, latent_dim, batch_first=True)

    def forward(self, x):
        _, h = self.rnn(x)
        return h.squeeze(0)


class Transition(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.mu = nn.Linear(latent_dim, input_dim)
        self.logvar = nn.Linear(latent_dim, input_dim)

    def forward(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        return mu, logvar


# ============================================================
# HAD-PCN Model
# ============================================================

class HADPCN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.transition = Transition(latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z0 = self.encoder(x)
        z_pred = self.transition(z0)
        mu, logvar = self.decoder(z_pred)
        return z0, z_pred, mu, logvar


# ============================================================
# Predictive Coding Refinement
# ============================================================

def refine_latent(z, x, model, steps=3, lr=0.01):
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([z], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        mu, logvar = model.decoder(z)
        recon_loss = ((x - mu) ** 2).mean()
        dyn_loss = ((z - model.transition(z)) ** 2).mean()
        loss = recon_loss + dyn_loss
        loss.backward()
        optimizer.step()

    return z.detach()


# ============================================================
# Loss
# ============================================================

def loss_fn(x, mu, logvar):
    return ((x - mu) ** 2).mean()


# ============================================================
# Training
# ============================================================

def train_model(model, data, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for x in data:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            z0, z_pred, mu, logvar = model(x)
            z_refined = refine_latent(z_pred, x.squeeze(0), model)

            mu, logvar = model.decoder(z_refined)
            loss = loss_fn(x.squeeze(0), mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss={total_loss:.4f}")


def reactive_score(x, model):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    _, _, mu, _ = model(x)
    return ((x.squeeze(0) - mu) ** 2).mean().item()


def proactive_score(x_seq, model, H=10):
    scores = []
    z = model.encoder(torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device))

    for _ in range(H):
        z = model.transition(z)
        mu, _ = model.decoder(z)
        score = ((mu) ** 2).mean().item()
        scores.append(score)

    return np.mean(scores)



# Main

def main():
    os.makedirs("outputs", exist_ok=True)

    train, test, labels = simulate_data()

    model = HADPCN(input_dim=10, latent_dim=16).to(device)

    train_model(model, train)

    reactive_train = [reactive_score(x, model) for x in train]
    reactive_test = [reactive_score(x, model) for x in test]

    proactive_train = [proactive_score(x, model) for x in train]
    proactive_test = [proactive_score(x, model) for x in test]

    np.save("outputs/reactive_train.npy", reactive_train)
    np.save("outputs/reactive_test.npy", reactive_test)
    np.save("outputs/proactive_train.npy", proactive_train)
    np.save("outputs/proactive_test.npy", proactive_test)
    np.save("outputs/test_labels.npy", labels)

    print("Saved outputs!")


if __name__ == "__main__":
    main()
