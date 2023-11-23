from pathlib import Path

import torch
import torch.nn as nn
import yaml


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def save_model(self, file_path=None):
        if file_path is None:
            base_path = Path(self.config.get('models', 'models'))
            # trial_number = self.config.get('trial_number', 1)
            trial_number = self.config.get('trial_number')
            rnn_name = self.config.get('rnn', 'default_rnn')
            file_path = base_path / f"trial_{trial_number}" / f"{rnn_name}.pth"
        else:
            file_path = Path(file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {'model_state_dict': self.state_dict(),
                     'config': self.config}
        if hasattr(self, 'optimizer'):
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(save_dict, file_path)

    @staticmethod
    def load_model(model_class, filename, device, config_path=None):
        checkpoint = torch.load(filename, map_location=device)

        if config_path:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = checkpoint['config']

        model = model_class(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and hasattr(model, 'optimizer'):
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)
        return model
