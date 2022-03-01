import argparse
import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import Encoder, Decoder, Discriminator
from dataset import get_dataset
from distributions import get_distribution
from utils import save_latent_variable, concat_image, save_image


def train(encoder, decoder, discriminator, opt):
    encode_optimizer = Adam(encoder.parameters(), lr=0.001)
    decode_optimizer = Adam(decoder.parameters(), lr=0.001)
    discriminate_optimizer = Adam(discriminator.parameters(), lr=0.001)

    image_dataset = get_dataset(opt.data_name, opt.data_root, opt.image_size, train=True)
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, shuffle=True)

    dist_dataset = get_distribution(opt.distribution, len(image_dataset), opt.num_classes)
    dist_loader = DataLoader(dist_dataset, batch_size=opt.batch_size, shuffle=True, worker_init_fn=lambda x: np.random.seed())

    encoder.train()
    decoder.train()
    discriminator.train()
    for i, ((image, image_label), (z_real, z_label)) in enumerate(zip(image_loader, dist_loader)):
        image = image.to(opt.device)
        image_label = image_label.to(opt.device)

        z_real = z_real.to(opt.device)
        z_label = z_label.to(opt.device)

        # reconstruct
        encode_optimizer.zero_grad()
        decode_optimizer.zero_grad()
        output = decoder(encoder(image))
        reconstruct_loss = -torch.mean(image * torch.log(output + 1e-8) + (1 - image) * torch.log(1 - output + 1e-8))

        reconstruct_loss.backward()
        encode_optimizer.step()
        decode_optimizer.step()

        # discriminator
        encode_optimizer.zero_grad()
        discriminate_optimizer.zero_grad()

        with torch.no_grad():
            z_fake = encoder(image)

        d_real = discriminator(z_real, z_label)
        d_fake = discriminator(z_fake, image_label)
        d_loss = -0.02 * torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))

        d_loss.backward()
        discriminate_optimizer.step()

        # encoder
        encode_optimizer.zero_grad()
        discriminate_optimizer.zero_grad()

        z_fake = encoder(image)
        e_fake = discriminator(z_fake, image_label)
        e_loss = -0.02 * torch.mean(torch.log(e_fake + 1e-8))

        e_loss.backward()
        encode_optimizer.step()

    return reconstruct_loss, e_loss, d_loss


def eval_encoder(file_name, encoder, opt):
    eval_dataset = get_dataset(opt.data_name, opt.data_root, opt.image_size, train=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    zs, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(eval_loader):
            image = image.to(opt.device)
            z = encoder(image)
            z = z.cpu().tolist()
            label = label.tolist()
            zs.extend(z)
            labels.extend(label)

    zs, labels = np.array(zs), np.array(labels)
    save_latent_variable(file_name, zs, labels)


def eval_decoder(file_name, decoder, opt):
    decoder.eval()
    with torch.no_grad():
        x, y = torch.meshgrid([torch.linspace(-3, 3, 8), torch.linspace(-3, 3, 8)], indexing="ij")
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        z = torch.cat((x, y), dim=1).to(opt.device)

        output = decoder(z)
        output = output.permute(0, 2, 3, 1).cpu().numpy()

    output = concat_image(output)
    save_image(file_name, output)


def main():
    parser = argparse.ArgumentParser(description="AAE")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="mnist")
    parser.add_argument("--distribution", type=str, default="gaussian")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=10)
    opt = parser.parse_args()

    os.makedirs("./outputs/encode", exist_ok=True)
    os.makedirs("./outputs/decode", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)


    encoder = Encoder(opt.image_size, opt.image_channels, opt.latent_dim).to(opt.device)
    decoder = Decoder(opt.image_size, opt.image_channels, opt.latent_dim).to(opt.device)
    discriminator = Discriminator(opt.latent_dim, opt.num_classes, True).to(opt.device)

    for epoch in range(opt.num_epochs):
        reconstruct_loss, e_loss, d_loss = train(encoder, decoder, discriminator, opt)
        print("reconstruct loss: {:.4f} encorder loss: {:.4f} discriminator loss: {:.4f}".format(reconstruct_loss, e_loss, d_loss))
        eval_encoder("./outputs/encode/{}.jpg".format(epoch), encoder, opt)
        eval_decoder("./outputs/decode/{}.jpg".format(epoch), decoder, opt)

    torch.save(encoder.state_dict(), "./weights/encoder.pth")
    torch.save(decoder.state_dict(), "./weights/decoder.pth")
    torch.save(discriminator.state_dict(), "./weights/discriminator.pth")


if __name__ == "__main__":
    main()
