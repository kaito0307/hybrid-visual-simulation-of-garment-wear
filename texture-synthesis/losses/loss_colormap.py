import torch
import numpy as np

from losses.appearance_loss_colormap import AppearanceLoss

class Loss(torch.nn.Module):
    def __init__(self, appearance_loss_weight=0.0, overflow_loss_weight=0.0,
                 appearance_loss_kwargs=None):
        super(Loss, self).__init__()

        self.appearance_loss_weight = appearance_loss_weight
        self.overflow_loss_weight = overflow_loss_weight

        self.appearance_loss_kwargs = {} if appearance_loss_kwargs is None else appearance_loss_kwargs

        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = torch.nn.ModuleDict()
        self.loss_weights = {}

        if self.appearance_loss_weight != 0:
            self.loss_mapper['appearance'] = AppearanceLoss(**self.appearance_loss_kwargs)
            self.loss_weights['appearance'] = self.appearance_loss_weight

        if self.overflow_loss_weight != 0:
            # Register the overflow loss as a function
            self.loss_weights['overflow'] = self.overflow_loss_weight

        self.appearance_loss_history = []

    def update_loss_weights(self, loss_log, epoch):
        # Look for 'appearance' or its detailed keys in the loss log
        app_loss_key = 'appearance'
        if app_loss_key in loss_log:
            self.appearance_loss_history.append(loss_log[app_loss_key])

    def get_overflow_loss(self, input_dict, return_summary=True):
        nca_state = input_dict['nca_state']
        # Penalize state values outside [-1.0, 1.0]
        overflow_loss = (nca_state - nca_state.clamp(-1.0, 1.0)).abs().mean()
        return overflow_loss, None, None

    def forward(self, input_dict, return_log=True, return_summary=True):
        loss = 0.0
        loss_log_dict = {}
        summary_dict = {}

        # 1. Losses in the ModuleDict (Appearance, Image, etc.)
        for loss_name, loss_module in self.loss_mapper.items():
            l, loss_log, sub_summary = loss_module(input_dict, return_summary=return_summary)

            if loss_log is not None:
                for sub_loss_name in loss_log:
                    # Log the detailed losses (e.g., per channel)
                    loss_log_dict[f'{loss_name}-{sub_loss_name}'] = loss_log[sub_loss_name].item() if torch.is_tensor(loss_log[sub_loss_name]) else loss_log[sub_loss_name]

            if sub_summary is not None:
                summary_dict.update(sub_summary)

            loss_log_dict[loss_name] = l.item()
            loss += l * self.loss_weights[loss_name]

        # 2. Function-based losses such as overflow
        if self.overflow_loss_weight != 0:
            l_over, _, _ = self.get_overflow_loss(input_dict)
            loss_log_dict['overflow'] = l_over.item()
            loss += l_over * self.loss_weights['overflow']

        output = [loss]
        if return_log:
            output.append(loss_log_dict)
        else:
            output.append(None)

        if return_summary:
            output.append(summary_dict)
        else:
            output.append(None)

        return output

if __name__ == "__main__":
    # Test code
    appearance_loss_kwargs = {
        "target_image_size": (128, 128),
        "device": "cpu"
    }
    loss_fn = Loss(appearance_loss_weight=1.0, overflow_loss_weight=0.1, 
                  appearance_loss_kwargs=appearance_loss_kwargs)

    # Dummy data
    input_dict = {
        'rendered_images': torch.rand(1, 3, 128, 128),
        'target_images': torch.rand(1, 3, 128, 128),
        'nca_state': torch.randn(1, 16, 32, 32)
    }
    
    l_total, l_log, s_dict = loss_fn(input_dict)
    print(f"Total Loss: {l_total.item()}")
    print(f"Log: {l_log}")