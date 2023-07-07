import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights


# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

data = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

# initialize gen and disc/critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(disc)


# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_disc = optim.RMSprop(disc.parameters(), lr=LEARNING_RATE)


# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()


for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train D: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        # loss_disc_real = nn.BCEWithLogitsLoss()(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        # loss_disc_fake = nn.BCEWithLogitsLoss()(disc_fake, torch.zeros_like(disc_fake))
        # loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # clip critic weights between -0.01, 0.01
        for p in disc.parameters():
            p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
        
        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = disc(fake).reshape(-1)
        # loss_gen = nn.BCEWithLogitsLoss()(gen_fake, torch.ones_like(gen_fake))
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            disc.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
            disc.train()

