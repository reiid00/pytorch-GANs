import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# CNN Architecture

class Discriminator(nn.Module):
    def __init__(self, channels_img=1, out_features=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, out_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_features, out_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_features * 2, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img=1, out_features=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, out_features * 4, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(out_features * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(out_features * 4, out_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_features * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(out_features * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
z_dim = 128
channels_img = 1
batch_size = 32
num_epochs = 50

disc = Discriminator(channels_img).to(device)
gen = Generator(z_dim, channels_img).to(device)
# Changed dims to ConvTranpose2d layer, 4-dim tensor
fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
transforms = transforms.Compose(
    [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):

        # No need to reshape the real image as it is passed to a CNN layer
        real = real.to(device)
        batch_size = real.shape[0]
        
        # 4-dimensional labels to match the output shape of the discriminator
        real_label = torch.full((batch_size, 1, 1, 1), 1.0, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1, 1, 1), 0.0, device=device, dtype=torch.float32)

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        
        # Changed dims to ConvTranpose2d layer, 4-dim tensor    
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real)
        lossD_real = criterion(disc_real, real_label)
        disc_fake = disc(fake.detach())
        lossD_fake = criterion(disc_fake, fake_label)

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max (log(D(G(z))))
        output = disc(fake)
        lossG = criterion(output, real_label)

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                data = real
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1


# Changes:

# Increase Network Size, in both Generator and Discriminator
# BatchNorm Layers
# Learning Rate to 1e-4
# Architecture of Generator and Discriminator to CNN (Conv2d and ConvTranspose2d)