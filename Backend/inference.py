import argparse
import os
import numpy as np
import PIL.Image as Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import torch

from denoisator import MangaDenoiser
from colorizator import MangaColorizator
from upscalator import MangaUpscaler
from utils.utils import distance_from_grayscale, save_image, clear_torch_cache


def process_image_safe(image_path, output_folder, config):
    try:
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")

        # Skip very small images
        if image.size[0] < 64 or image.size[1] < 64:
            print(f"[!] {image_name} is too small, skipping.")
            return

        image = np.array(image)

        coloredness = distance_from_grayscale(image)
        if coloredness > 1:
            print(f"[+] {image_name} is already colored, skipping.")
            return

        # Initialize models inside thread (safe from race conditions)
        denoiser = MangaDenoiser(config) if config.denoise else None
        colorizer = MangaColorizator(config) if config.colorize else None
        upscaler = MangaUpscaler(config) if config.upscale else None

        if config.denoise:
            print(f"[*] Denoising {image_name}...")
            image = denoiser.denoise(image, config.denoise_sigma)

        if config.colorize:
            print(f"[*] Colorizing {image_name}...")
            image = image.astype(np.float32) / 255.0
            colorizer.set_image(image, config.colorized_image_size)
            image = colorizer.colorize()
            image = (image * 255).clip(0, 255).astype(np.uint8)

        if config.upscale:
            print(f"[*] Upscaling {image_name} by {config.upscale_factor}x...")
            image = image.astype(np.float32) / 255.0
            image = upscaler.upscale(image, config.upscale_factor)
            image = (image * 255).clip(0, 255).astype(np.uint8)

        output_path = os.path.join(output_folder, image_name)
        save_image(image, output_path)
        print(f"[+] Processed {image_name} -> Saved to {output_path}")

        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Batch Colorize Images")
    parser.add_argument("--input_path", type=str, default="input", help="Folder containing images")
    parser.add_argument("--output_path", type=str, default="output", help="Folder to save processed images")

    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Device to use')

    parser.add_argument('--colorizer_path', default='networks/generator.zip')
    parser.add_argument('--extractor_path', default='networks/extractor.pth')
    parser.add_argument('--upscaler_path', default='networks/RealESRGAN_x4plus_anime_6B.pt')
    parser.add_argument('--upscaler_type', choices=['ESRGAN', 'GigaGAN'], default='ESRGAN')

    parser.add_argument('--no-upscale', dest='upscale', action='store_false', default=True, help='Disable upscaling')
    parser.add_argument('--no-colorize', dest='colorize', action='store_false', default=True, help='Disable colorization')
    parser.add_argument('--no-denoise', dest='denoise', action='store_false', default=True, help='Disable denoiser')
    parser.add_argument('--upscale_factor', choices=[2, 4], default=4, type=int, help='Upscale by x2 or x4')
    parser.add_argument('--denoise_sigma', default=25, type=int, help='How much noise to expect from the image')

    parser.add_argument('--threads', default=1, type=int, help='Number of threads for parallel processing')

    config = parser.parse_args()
    os.makedirs(config.output_path, exist_ok=True)

    config.upscaler_tile_size = 256
    config.colorizer_tile_size = 0
    config.tile_pad = 8
    config.colorized_image_size = 576

    images = [f for f in os.listdir(config.input_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    image_paths = [os.path.join(config.input_path, f) for f in images]

    print(f"[*] Processing {len(image_paths)} images using {config.threads} threads...")

    with ThreadPoolExecutor(max_workers=config.threads) as executor:
        futures = {
            executor.submit(process_image_safe, path, config.output_path, config): path
            for path in image_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"[!] Exception during threaded processing: {exc}")

    print("[+] Batch processing complete")
    clear_torch_cache()
    print("[+] Components released")

if __name__ == "__main__":
    main()
