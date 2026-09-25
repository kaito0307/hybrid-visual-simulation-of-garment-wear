import os
import numpy as np
import torch
import yaml
import argparse
import wandb
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from losses.loss_colormap import Loss
from models.nca2d import NCA, NoiseNCA, PENCA
from models.siren import Siren
from utils.render import Renderer2D
from utils.misc import process_output_channels

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/nca2d/colormap.yaml', help="configuration")

def load_image_tensor(path, size, device):
    """Load an image and convert it to a tensor."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device)

def main(config):
    device = torch.device(config['device'])

    def get_device_config(cfg):
        cfg['device'] = device
        return cfg

    # --- Set up the models ---
    nca_kwargs = get_device_config(config['nca']['nca_kwargs'])
    nca_type = config['nca']['type']
    model = {"NCA": NCA, "NoiseNCA": NoiseNCA, "PENCA": PENCA}[nca_type](**nca_kwargs).to(device)
    
    total_channels, output_channels = process_output_channels(config['num_channels'])

    nca_output_type = config['nca']['output_type']
    nca_output_dim = model.channels
    if nca_output_type == "z":
        nca_output_dim *= model.perception_kernels
    
    # SIREN (decoder)
    siren = Siren(in_features=nca_output_dim, coord_dim=2, out_features=total_channels, **config['siren']).to(device)

    print(f"NCA Channels: {model.channels} (Last 3 reserved for Colormap)")
    print(f"NCA Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Siren Parameters: {sum(p.numel() for p in siren.parameters())}")
    
    # --- Load the data (pairs for wear levels 1-10) ---
    target_size = config['loss']['appearance_loss_kwargs']['target_image_size']
    textures = []
    colormaps = []
    
    tex_dir = config['train']['texture_path']
    map_dir = config['train']['colormap_path']
    
    print(f"Loading datasets from {tex_dir} and {map_dir}...")
    for i in range(1, 11):
        # Texture (e.g., 1.jpg) and wear map (e.g., 1_colormap.png)
        tex_path = os.path.join(tex_dir, f"{i}.jpg")
        map_path = os.path.join(map_dir, f"{i}_colormap.png")
        
        textures.append(load_image_tensor(tex_path, target_size, device))
        colormaps.append(load_image_tensor(map_path, target_size, device))
    
    textures = torch.cat(textures, dim=0)   # [10, 3, H, W]
    colormaps = torch.cat(colormaps, dim=0) # [10, 3, H, W]

    # --- Prepare for training ---
    with torch.no_grad():
        grid_size = config['train']['nca_grid_size']
        pool = model.seed(config['train']['pool_size'], grid_size[0], grid_size[1])

        config['loss']['appearance_loss_kwargs']['total_channels'] = total_channels
        config['loss']['appearance_loss_kwargs']['output_channels'] = output_channels
        loss_fn = Loss(**config['loss'])
        renderer = Renderer2D(**config['renderer'])

    parameters = list(model.parameters()) + list(siren.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config['train']['lr'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                       milestones=config['train']['lr_decay_steps'],
                                                       gamma=config['train']['lr_decay_gamma'])

    step_range = config['train']['step_range']
    inject_seed_interval = config['train']['inject_seed_interval']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']

    wandb_enabled = 'wandb' in config
    if wandb_enabled:
        wandb.login(key=config['wandb']['key'], relogin=True)
        wandb.init(project=config['wandb']['project'], name=config['experiment_name'],
                   dir=config['experiment_path'], config=config)
    
    # --- Training loop ---
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            batch_idx = np.random.choice(len(pool), batch_size, replace=False)
            x = pool[batch_idx]
            
            # Randomly pick a wear level from 1 to 10
            target_idx = np.random.randint(0, 10)
            target_tex = textures[target_idx:target_idx+1].repeat(batch_size, 1, 1, 1)
            target_map = colormaps[target_idx:target_idx+1].repeat(batch_size, 1, 1, 1)

            if epoch % inject_seed_interval == 0:
                x[:1] = model.seed(1, grid_size[0], grid_size[1])

        # [Conditioning] Inject the wear map into the NCA state
        # The last 3 channels are fixed to the wear map color
        x[:, -3:] = target_map

        step_n = np.random.randint(step_range[0], step_range[1])
        x0 = x.clone()
        
        # Run the NCA steps
        for _ in range(step_n):
            x, z = model(x)
            # Re-inject the wear map at every step (so that it does not diffuse)
            x[:, -3:] = target_map

        # Render
        x_render = x if nca_output_type == "s" else z
        rendered_image = renderer.render(x_render.permute(0, 2, 3, 1), siren, None, fs_shader="vanilla")
        rendered_image = rendered_image.permute(0, 3, 1, 2)

        # Compute the loss
        input_dict = {
            'rendered_images': rendered_image,
            'target_images': target_tex,
            'nca_state': x,
            'step_n': step_n,
        }
        
        return_summary = (epoch % config['train']['summary_interval'] == 0)
        loss, loss_log, summary = loss_fn(input_dict, return_summary=return_summary)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            # Normalize the gradients to prevent them from exploding
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= (p.grad.norm() + 1e-8)
            optimizer.step()
            lr_scheduler.step()
            
            # Update the pool (the wear map channels are kept)
            pool[batch_idx] = x
            loss_fn.update_loss_weights(loss_log, epoch)

        # Logging
        if wandb_enabled:
            wandb.log(loss_log, step=epoch)
            if return_summary:
                wandb.log({'Rendered': wandb.Image(summary['appearance-images'], 
                           caption=f'Level {target_idx+1}')}, step=epoch)

    # --- Save ---
    torch.save(model.state_dict(), f'{config["experiment_path"]}/model.pth')
    torch.save(siren.state_dict(), f'{config["experiment_path"]}/siren.pth')
    if wandb_enabled: wandb.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if 'experiment_path' not in config or config['experiment_path'] is None:
        # If experiment_path is not set
        exp_name = config.get('experiment_name', 'default_experiment')
        
        # Get result_path from the config (defaults to "results")
        base_result_path = config.get('result_path', 'results')
        
        # Use {result_path}/{exp_name}
        config['experiment_path'] = os.path.join(base_result_path, exp_name)
    
    exp_path = config['experiment_path']
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    yaml.dump(config, open(f'{exp_path}/config.yaml', 'w'))

    main(config)