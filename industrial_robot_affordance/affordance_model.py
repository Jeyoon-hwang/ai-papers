"""
Industrial Affordance Model
VAE + Affordance Detection Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np


class SimpleVAE(nn.Module):
    """
    간단한 VAE (게임 모델 대체)
    실제 구현에서는 pretrained model 로드
    """
    
    def __init__(self, latent_dim: int = 64, input_channels: int = 3):
        """
        Initialize VAE
        
        Args:
            latent_dim: latent vector 차원
            input_channels: 입력 이미지 채널 수 (RGB=3)
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: Conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: Transposed Conv
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution
        
        Args:
            x: (batch_size, C, H, W)
            
        Returns:
            (mu, logvar): latent distribution parameters
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: mean of latent distribution
            logvar: log variance
            
        Returns:
            z: sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image
        
        Args:
            z: (batch_size, latent_dim)
            
        Returns:
            reconstructed image
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, 4, 4)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            (recon_x, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class IndustrialAffordanceHead(nn.Module):
    """
    산업용 affordance 감지 헤드
    VAE latent vector → 6개 affordance
    """
    
    def __init__(self, input_dim: int = 64, num_affordances: int = 6):
        """
        Initialize affordance head
        
        Args:
            input_dim: VAE latent dimension
            num_affordances: 6 affordances
        """
        super().__init__()
        
        self.affordance_names = [
            'graspable',
            'stackable',
            'insertable',
            'placeable',
            'moveable',
            'fragile'
        ]
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_affordances),
            nn.Sigmoid()  # Output [0, 1]
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict affordances from latent vector
        
        Args:
            latent: (batch_size, input_dim)
            
        Returns:
            affordances: (batch_size, num_affordances)
        """
        return self.head(latent)


class IndustrialAffordanceModel(nn.Module):
    """
    Full pipeline: Image → Affordances
    """
    
    def __init__(self, latent_dim: int = 64):
        """
        Initialize full model
        """
        super().__init__()
        self.vae = SimpleVAE(latent_dim=latent_dim)
        self.affordance_head = IndustrialAffordanceHead(
            input_dim=latent_dim,
            num_affordances=6
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Image → Affordances
        
        Args:
            x: (batch_size, 3, H, W) normalized to [0, 1]
            
        Returns:
            {
                'affordances': (batch_size, 6),
                'latent': (batch_size, latent_dim),
                'recon': (batch_size, 3, H, W)
            }
        """
        # Encode image
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        
        # Predict affordances
        affordances = self.affordance_head(z)
        
        # Reconstruct image
        recon_x = self.vae.decode(z)
        
        return {
            'affordances': affordances,
            'latent': z,
            'recon': recon_x,
            'mu': mu,
            'logvar': logvar
        }
    
    def predict_affordances(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Predict affordances for single image (inference mode)
        
        Args:
            x: (1, 3, H, W)
            
        Returns:
            affordances dict: {'graspable': 0.9, ...}
        """
        with torch.no_grad():
            output = self.forward(x)
            aff_tensor = output['affordances'].squeeze(0)
            
            affordances_dict = {
                name: float(aff_tensor[i])
                for i, name in enumerate(self.affordance_head.affordance_names)
            }
            
            return affordances_dict


class AffordanceTrainer:
    """
    Affordance 모델 학습
    """
    
    def __init__(self, model: IndustrialAffordanceModel, device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: Industrial affordance model
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def vae_loss(self, recon_x, x, mu, logvar, beta=0.1):
        """
        VAE loss = Reconstruction + KL divergence
        
        Args:
            recon_x: reconstructed image
            x: original image
            mu: latent mean
            logvar: latent log variance
            beta: KL weight
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(recon_x, x)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)
        
        return recon_loss + beta * kl_loss
    
    def affordance_loss(self, pred_aff, target_aff):
        """
        Affordance prediction loss (binary cross entropy)
        """
        return self.bce_loss(pred_aff, target_aff)
    
    def train_step(self,
                  images: torch.Tensor,
                  affordances: torch.Tensor,
                  optimizer,
                  vae_beta=0.1,
                  aff_weight=1.0):
        """
        Single training step
        
        Args:
            images: (batch_size, 3, H, W)
            affordances: (batch_size, 6)
            optimizer: torch optimizer
            vae_beta: VAE KL weight
            aff_weight: affordance loss weight
        """
        self.model.train()
        
        output = self.model(images)
        
        # VAE loss
        vae_loss = self.vae_loss(
            output['recon'],
            images,
            output['mu'],
            output['logvar'],
            beta=vae_beta
        )
        
        # Affordance loss
        aff_loss = self.affordance_loss(output['affordances'], affordances)
        
        # Total loss
        total_loss = vae_loss + aff_weight * aff_loss
        
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'vae_loss': vae_loss.item(),
            'aff_loss': aff_loss.item()
        }
    
    def validate(self,
                images: torch.Tensor,
                affordances: torch.Tensor,
                vae_beta=0.1,
                aff_weight=1.0):
        """
        Validation step
        """
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(images)
            
            vae_loss = self.vae_loss(
                output['recon'],
                images,
                output['mu'],
                output['logvar'],
                beta=vae_beta
            )
            
            aff_loss = self.affordance_loss(output['affordances'], affordances)
            total_loss = vae_loss + aff_weight * aff_loss
            
            # Affordance accuracy (per-class)
            pred_aff = (output['affordances'] > 0.5).float()
            target_aff = (affordances > 0.5).float()
            aff_accuracy = (pred_aff == target_aff).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'vae_loss': vae_loss.item(),
            'aff_loss': aff_loss.item(),
            'aff_accuracy': aff_accuracy.item()
        }


# 테스트
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 모델 생성
    model = IndustrialAffordanceModel(latent_dim=64)
    model.to(device)
    
    # 테스트 입력
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128).to(device)  # 이미지
    target_aff = torch.rand(batch_size, 6).to(device)      # affordances
    
    # Forward pass
    output = model(x)
    print("\n=== Model Output ===")
    print(f"Affordances shape: {output['affordances'].shape}")
    print(f"Latent shape: {output['latent'].shape}")
    print(f"Reconstruction shape: {output['recon'].shape}")
    
    # 예제 affordances
    print("\n=== Sample Affordances ===")
    pred_aff = output['affordances'][0].detach().cpu().numpy()
    affordance_names = model.affordance_head.affordance_names
    for name, value in zip(affordance_names, pred_aff):
        print(f"{name:15s}: {value:.3f}")
    
    # 학습
    print("\n=== Training Test ===")
    trainer = AffordanceTrainer(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 1 step 학습
    loss_dict = trainer.train_step(x, target_aff, optimizer)
    print(f"Training losses: {loss_dict}")
    
    # Validation
    val_loss = trainer.validate(x, target_aff)
    print(f"Validation metrics: {val_loss}")
    
    print("\n✓ Model test complete!")
