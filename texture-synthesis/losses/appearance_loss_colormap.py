import torch
import torchvision.models as torch_models
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

class AppearanceLoss(torch.nn.Module):
    def __init__(self,
                 target_image_size=(256, 256),
                 patch_size=(256, 256),
                 num_scales=1,
                 relative_scale=1,
                 total_channels=3,
                 output_channels={"RGB": [0, 1, 2]},
                 vgg_layers=(1, 6, 11, 18, 25),
                 max_image_size=(2048, 2048),
                 device='cuda:0',
                 **kwargs): # Accept and ignore unused arguments
        super(AppearanceLoss, self).__init__()

        self.target_image_size = target_image_size
        self.patch_size = patch_size
        self.num_scales = num_scales
        self.relative_scale = relative_scale
        self.max_image_size = max_image_size
        self.total_channels = total_channels
        self.output_channels = output_channels
        self.vgg_layers = vgg_layers
        self.device = device

        vgg = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        self.vgg = torch.nn.Sequential(*list(vgg.children()))
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.style_loss_fn = OptimalTransportLoss(n_samples=1024, device=self.device)

    def get_vgg_features(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        x = (x - mean) / std

        features = []
        for i, layer in enumerate(self.vgg[:max(self.vgg_layers) + 1]):
            x = layer(x)
            if i in self.vgg_layers:
                features.append(x)
        return features

    def forward(self, input_dict, return_summary=True):
        rendered_images = input_dict['rendered_images']
        target_images_full = input_dict['target_images']
        
        batch_size, channels, _, _ = rendered_images.shape
        loss_total = 0.0
        loss_log = {}

        # Compute the sizes of the target and generated images (taking relative_scale into account)
        # sx, sy are the resolution of the crop taken from the target image
        sx = int(self.patch_size[0] * 2 ** (self.relative_scale - 1))
        sy = int(self.patch_size[1] * 2 ** (self.relative_scale - 1))

        for scale in range(self.num_scales):
            xs = []
            ts = []
            
            for key in sorted(self.output_channels):
                oc = self.output_channels[key]
                x = rendered_images[:, oc]
                if len(oc) == 1: x = x.repeat(1, 3, 1, 1)
                
                # Adjust the size for each scale
                curr_sx, curr_sy = sx * (2**scale), sy * (2**scale)
                curr_px, curr_py = self.patch_size[0] * (2**scale), self.patch_size[1] * (2**scale)

                # Resize the generated image to the patch size
                x_patch = F.interpolate(x, size=(curr_px, curr_py), mode='bilinear', align_corners=False)
                xs.append(x_patch)

                # Resize/crop the target image according to relative_scale
                t = F.interpolate(target_images_full, size=(curr_sx, curr_sy), mode='bilinear', align_corners=False)
                ts.append(t)
            
            gen_batch = torch.cat(xs, dim=0)
            target_batch = torch.cat(ts, dim=0)

            gen_features = self.get_vgg_features(gen_batch)
            target_features = self.get_vgg_features(target_batch)

            current_loss = self.style_loss_fn(target_features, gen_features)
            loss_total += current_loss.mean()
            
            for i, key in enumerate(sorted(self.output_channels)):
                loss_log[f"{key}_scale_{scale}"] = current_loss[i]

        loss_final = loss_total / self.num_scales
        
        summary = None
        if return_summary:
            with torch.no_grad():
                vis_gen = F.interpolate(rendered_images[:, self.output_channels[sorted(self.output_channels.keys())[0]]][0:1], size=self.target_image_size)
                if vis_gen.shape[1] == 1: vis_gen = vis_gen.repeat(1, 3, 1, 1)
                vis_target = F.interpolate(target_images_full[0:1], size=self.target_image_size)
                vis_stack = torch.cat([vis_target, vis_gen], dim=3)
                vis_stack = torch.clamp(vis_stack, 0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
                summary = {"appearance-images": Image.fromarray((vis_stack * 255).astype(np.uint8))}

        return loss_final, loss_log, summary

class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, n_samples=1024, device='cuda:0'):
        super().__init__()
        self.n_samples = n_samples
        self.device = device
        self.color_transform = torch.tensor(
            [[0.577350, 0.577350, 0.577350],
             [-0.577350, 0.788675, -0.211325],
             [-0.577350, -0.211325, 0.788675]], device=device, requires_grad=False)

    def rgb_to_yuv(self, rgb):
        return torch.einsum('bnc,ck->bnk', rgb, self.color_transform)

    def color_matching_loss(self, x, y):
        x_yuv = self.rgb_to_yuv(x)
        y_yuv = self.rgb_to_yuv(y)
        pairwise_distance = self.pairwise_distances_l2(x_yuv, y_yuv) + self.pairwise_distances_cos(x_yuv, y_yuv)
        m1, _ = pairwise_distance.min(1)
        m2, _ = pairwise_distance.min(2)
        return torch.max(m1.mean(dim=1), m2.mean(dim=1))

    @staticmethod
    def pairwise_distances_l2(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True) ** 2
        y_t = y.transpose(1, 2)
        y_norm = torch.norm(y_t, dim=1, keepdim=True) ** 2
        cross = torch.matmul(x, y_t)
        dist = x_norm + y_norm - 2.0 * cross
        return torch.clamp(dist, 1e-5, 1e5) / x.size(2)

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)
        y_t = y.transpose(1, 2)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)
        return 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)

    @staticmethod
    def style_loss(x, y):
        pairwise_distance = OptimalTransportLoss.pairwise_distances_cos(x, y)
        m1, _ = pairwise_distance.min(1)
        m2, _ = pairwise_distance.min(2)
        return torch.max(m1.mean(dim=1), m2.mean(dim=1))

    @staticmethod
    def moment_loss(x, y):
        mu_x = torch.mean(x, 1, keepdim=True)
        mu_y = torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))
        x_c = x - mu_x
        y_c = y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)
        return mu_diff + torch.abs(x_cov - y_cov).mean(dim=(1, 2))

    def forward(self, target_features, generated_features):
        loss = 0.0
        for i, (y, x) in enumerate(zip(target_features, generated_features)):
            b, c_y, h_y, w_y = y.shape
            _, c_x, h_x, w_x = x.shape
            
            # Flatten
            x = x.view(b, c_x, -1)
            y = y.view(b, c_y, -1)
            
            n_samples = min(x.shape[-1], y.shape[-1], self.n_samples)
            
            # Random sampling
            idx_x = torch.randperm(x.shape[-1], device=x.device)[:n_samples]
            idx_y = torch.randperm(y.shape[-1], device=y.device)[:n_samples]
            
            x_s = x[:, :, idx_x].transpose(1, 2) # [B, N, C]
            y_s = y[:, :, idx_y].transpose(1, 2) # [B, N, C]

            # Use color matching only for the first comparison with 3 channels (RGB)
            if c_x == 3 and i == 0:
                loss += self.color_matching_loss(x_s, y_s)
            else:
                # Otherwise (e.g., VGG features), use the style loss and the moment loss
                loss += self.style_loss(x_s, y_s) + self.moment_loss(x_s, y_s)
            
        return loss