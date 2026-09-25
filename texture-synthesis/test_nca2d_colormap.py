import os
import numpy as np
import torch
import yaml
import argparse
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models.nca2d import NCA, NoiseNCA, PENCA
from utils.misc import process_output_channels
from utils.render import Renderer2D, Siren

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/nca2d/colormap.yaml', help="configuration")

def main(config):
    device = torch.device(config['device'])

    # --- Get the wear map path from the config ---
    # Read the 'rgb' entry from the nested config
    target_dict = config['loss']['appearance_loss_kwargs']['target_images_path']
    colormap_path = target_dict.get('rgb', 'data/colormaps_each_fabrics/wool_knit/jumping/00130_2000x4000_blurred.png')

    def get_device_config(cfg):
        cfg['device'] = device
        return cfg

    with torch.no_grad():
        # --- Initialize the models ---
        nca_kwargs = get_device_config(config['nca']['nca_kwargs'])
        nca_type = config['nca']['type']
        model = {"NCA": NCA, "NoiseNCA": NoiseNCA, "PENCA": PENCA}[nca_type](**nca_kwargs).to(device)
        total_channels, output_channels = process_output_channels(config['num_channels'])

        nca_output_type = config['nca'].get('output_type', "s")
        nca_output_dim = model.channels
        if nca_output_type == "z":
            nca_output_dim *= model.perception_kernels
            
        siren = Siren(in_features=nca_output_dim, coord_dim=2, out_features=total_channels, **config['siren']).to(device)

        # --- Load the weights ---
        model_path = os.path.join(config["experiment_path"], 'model.pth')
        siren_path = os.path.join(config["experiment_path"], 'siren.pth')
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return

        model.load_state_dict(torch.load(model_path, map_location=device))
        siren.load_state_dict(torch.load(siren_path, map_location=device))
        model.eval()
        siren.eval()

        # --- Load the wear map and use its size as the output resolution ---
        if not os.path.exists(colormap_path):
            raise FileNotFoundError(f"Colormap not found at: {colormap_path}")
            
        cmap_img = Image.open(colormap_path).convert("RGB")
        W, H = cmap_img.size
        print(f"Loaded colormap: {colormap_path} (Size: {W}x{H})")
        
        target_map = TF.to_tensor(cmap_img).unsqueeze(0).to(device) # [1, 3, H, W]

        # --- Create the initial seed ---
        x = model.seed(1, H, W)
        x[:, -3:] = target_map

        renderer = Renderer2D(**config['renderer'])
        output_dir = os.path.join(config['experiment_path'], 'inference')
        os.makedirs(output_dir, exist_ok=True)
        cmap_name = os.path.basename(colormap_path).split('.')[0]

        # --- Synthesize the texture ---
        # Run the NCA until the pattern becomes stable
        print(f"Generating image at resolution {H}x{W}...")
        for _ in tqdm(range(600), desc="NCA Steps"):
            x, z = model(x)
            x[:, -3:] = target_map # Overwrite the wear map condition at every step

        # Render
        x_render = x if nca_output_type == "s" else z
        image = renderer.render(x_render.permute(0, 2, 3, 1), siren, target_channels=output_channels, fs_shader="vanilla")
        image_pil = Renderer2D.to_pil(image)
        
        save_path = os.path.join(output_dir, f"output_{cmap_name}_{H}x{W}.jpg")
        image_pil.save(save_path)
        print(f"Success! Image saved to: {save_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set the experiment path (the output directory used during training)
    exp_name = config.get('experiment_name', 'default')
    result_path = config.get('result_path', 'results')
    config['experiment_path'] = os.path.join(result_path, exp_name)

    main(config)