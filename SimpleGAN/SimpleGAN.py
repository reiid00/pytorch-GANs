import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# ToDo: Improve GAN

# 1. Larger Network
# 2. Normalization with BatchNorm
# 3. Different Learning Rate
# 4. Change architecture to CNN

class Discriminator(nn.Module):

    def __init__(self, img_dim,out_features=128,act_val=0.1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, out_features),
            nn.LeakyReLU(act_val),
            nn.Linear(out_features,1), # linear to 1 dim, 0 or 1, real or fake, equivalently
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):

    def __init__(self, z_dim, img_dim, out_features = 256, act_val=0.1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, out_features),
            nn.LeakyReLU(act_val),
            nn.Linear(out_features, img_dim),
            nn.Tanh() # Output of pixel values are between -1 and 1, input will be normalized to be between (-1,1)
        )
    def forward(self, x):
        return self.gen(x)

# Hyper Parameters

# GANs are highly sensitive to HyperParameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4 # Best LR for Adam according to Andrej Karpathy
z_dim = 128
img_dim = 28 * 28 * 1 # mnist
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
mean_mnist = 0.1307
deviation_mnist = 0.3081
transforms = transforms.Compose(
    [ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size= batch_size, shuffle=True)

opt_disc = optim.Adam (disc.parameters(), lr=lr) # Discriminator Optimizer
opt_gen = optim.Adam (gen.parameters(), lr=lr) # Generator Optimizer
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        assert batch_size == real.shape[0], f"Batch Size not being correctly structured"

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z)))

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) /2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max (log(D(G(z))))

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1