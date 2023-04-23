import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator, Generator, initialize_weights

# HyperParams
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 2e-4
batch_size = 128
img_size = 64
channels_img = 3
z_dim = 100
num_epochs = 20
features_dic = 64
features_gen = 64

transforms = transforms.Compose(
    [
    
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
    [0.5 for _ in range(channels_img)], 
    [0.5 for _ in range(channels_img)], 
    )
    ]
)

dataset = datasets.CelebA(root='data', split='train', transform=transforms, download=True)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(z_dim, channels_img, features_gen).to(device)
discriminator = Discriminator(channels_img, features_dic).to(device)

initialize_weights(generator)
initialize_weights(discriminator)

opt_generator = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

writer_fake = SummaryWriter(f"Collection/models/DCGAN/runs/DCGAN_CELEBA/fake")
writer_real = SummaryWriter(f"Collection/models/DCGUN/runs/DCGAN_CELEBA/real")
step = 0

generator.train()
discriminator.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = generator(noise)
        # Train Discriminator: max log(D(real)) + log(1 - D(G(z)))

        disc_real = discriminator(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = discriminator(fake.detach()).reshape(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) /2
        discriminator.zero_grad()
        lossD.backward()
        opt_discriminator.step()

        # Train Generator: min log(1 - D(G(z))) <-> max (log(D(G(z))))

        output = discriminator(fake).reshape(-1)
        lossG = criterion(output, torch.ones_like(output))

        generator.zero_grad()
        lossG.backward()
        opt_generator.step()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(fixed_noise)
                data = real
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "CelebA Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "CelebA Real Images", img_grid_real, global_step=step
                )
                step += 1
