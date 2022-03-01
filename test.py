import argparse
import torch
from models import Decoder
from utils import save_images


def main():
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=2)
    opt = parser.parse_args()

    decoder = Decoder(opt.image_size, opt.image_channels, opt.latent_dim).to(opt.device)
    decoder.load_state_dict(torch.load("./weights/decoder.pth"))
    decoder.eval()

    with torch.no_grad():
        z = torch.randn(1, opt.latent_dim).to(opt.device)
        output = decoder(z)
        output = output.permute(0, 2, 3, 1).cpu().numpy()
        save_images("./outputs", output)


if __name__ == "__main__":
    main()
