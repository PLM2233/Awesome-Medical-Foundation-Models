import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor


class Curia(nn.Module):
    def __init__(self, pretrained_path=None, freeze=False):
        super().__init__()
        
        repo_id = "raidium/curia"
        self.processor = AutoImageProcessor.from_pretrained(
            repo_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        self.encoder = AutoModel.from_pretrained(
            repo_id, trust_remote_code=True
        )
        self.fc = nn.Identity()
        
        if freeze:
            self.encoder.eval().requires_grad_(False)
    
    def forward(self, x):
        """
        Args:
            x: input image tensor or numpy array
        Returns:
            cls_tokens: [B, 768]
            patch_tokens: [B, N, 768] where N = num_patches
        """
        # Process image if numpy array
        if isinstance(x, np.ndarray):
            model_input = self.processor(x)["pixel_values"]
        else:
            model_input = x
        print(f"Model input shape: {model_input.shape}")  
        outputs = self.encoder(pixel_values=model_input, output_hidden_states=True)
        # cls_tokens = outputs.last_hidden_state[:, 0]  # [B, 768]
        # patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, N, 768]
        print("Encoder output shape:", outputs.last_hidden_state.shape)

        output = self.fc(outputs.last_hidden_state[:, 1:, :].mean(dim=1)) # 对patch tokens做平均池化

        return output


if __name__ == '__main__':
    model = Curia(freeze=False)
    print(model.processor)
    print(model.encoder)
    
    img =  np.random.rand(256, 256)   # single axial slice, in PL orientation
    output = model(img)
    
    print(f"Output shape: {output.shape}")
