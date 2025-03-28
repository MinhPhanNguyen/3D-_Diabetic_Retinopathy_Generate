import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from utils.config import config
from utils.logger import logger

class RayGenerator:
    """Generates rays for NeRF training and rendering"""
    def __init__(self, height, width, focal):
        self.height = height
        self.width = width
        self.focal = focal
        
        # Create coordinate grid
        i, j = torch.meshgrid(
            torch.linspace(0, width-1, width),
            torch.linspace(0, height-1, height),
            indexing='ij'
        )
        self.pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=-1)  # (W, H, 3)
        
    def generate_rays(self, pose, near=0.1, far=6.0, num_samples=64):
        """Generate rays from camera pose"""
        # Convert pose to tensor
        pose = torch.tensor(pose, dtype=torch.float32)
        
        # Camera center (translation component)
        cam_center = pose[:3, 3]
        
        # Ray directions (transform pixel coordinates)
        directions = (self.pixel_coords - torch.tensor([self.width/2, self.height/2, 0])) / self.focal
        directions = directions @ pose[:3, :3].T  # Rotate by camera rotation
        
        # Normalize directions
        directions = F.normalize(directions, p=2, dim=-1)
        
        # Create ray origins and directions
        rays_o = cam_center.expand_as(directions)
        rays_d = directions
        
        # Sample points along rays
        t_vals = torch.linspace(near, far, num_samples)
        points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
        
        return rays_o, rays_d, points, t_vals

class PositionalEncoding(nn.Module):
    """Positional encoding layer for NeRF inputs"""
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Create frequency bands
        self.freq_bands = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs)
        
    def forward(self, x):
        """
        Apply positional encoding to input
        Args:
            x: Input tensor of shape (..., 3) [x,y,z coordinates]
        Returns:
            encoded: Encoded tensor of shape (..., 3*(2*num_freqs + include_input))
        """
        # If no encoding, just return input
        if self.num_freqs == 0:
            return x
            
        # Scale coordinates by frequency bands
        scaled = x[..., None] * self.freq_bands  # (..., 3, num_freqs)
        
        # Compute sin and cos for each frequency
        sin_enc = torch.sin(scaled)  # (..., 3, num_freqs)
        cos_enc = torch.cos(scaled)  # (..., 3, num_freqs)
        
        # Concatenate sin and cos features
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # (..., 3, 2*num_freqs)
        
        # Flatten last two dimensions
        encoded = encoded.flatten(-2, -1)  # (..., 3*2*num_freqs)
        
        # Include original input if requested
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
            
        return encoded

    @property
    def output_dim(self):
        """Calculate the output dimension of the encoding"""
        return 3 * (2 * self.num_freqs + (1 if self.include_input else 0))

class NeRFWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        
        # Positional encoding configuration
        self.pos_enc = PositionalEncoding(num_freqs=10)
        
        # Network configuration
        self.network = self._build_network()
        self.to(self.device)
        
        # Ray generator
        self.ray_gen = RayGenerator(
            height=config.IMAGE_SIZE[0],
            width=config.IMAGE_SIZE[1],
            focal=500  # Adjust based on your camera
        )
        
    def _build_network(self):
        """Build the MLP network"""
        layers = []
        input_dim = self.pos_enc.output_dim
        
        # Hidden layers
        for _ in range(4):
            layers.append(nn.Linear(input_dim, 256))
            layers.append(nn.ReLU())
            input_dim = 256
            
        # Output layers
        layers.append(nn.Linear(256, 4))  # RGB (3) + density (1)
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        encoded = self.pos_enc(x)
        outputs = self.network(encoded)
        
        # Split into RGB and density
        rgb = torch.sigmoid(outputs[..., :3])  # RGB in [0,1]
        density = F.softplus(outputs[..., 3:])  # Density > 0
        
        return torch.cat([rgb, density], dim=-1)

    def train(self, images, poses, output_dir, num_iters=1000):
        """Training loop for NeRF"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        
        for i in range(num_iters):
            # Randomly select an image
            idx = np.random.randint(len(images))
            img = images[idx]
            pose = poses[idx]
            
            # Generate rays
            rays_o, rays_d, points, t_vals = self.ray_gen.generate_rays(pose)
            
            # Flatten and select random batch
            points = points.view(-1, 3)
            target_colors = img.view(-1, 3)
            
            # Random batch selection
            rand_idx = torch.randperm(points.shape[0])[:config.BATCH_SIZE]
            batch_points = points[rand_idx]
            batch_targets = target_colors[rand_idx]
            
            # Forward pass
            outputs = self(batch_points)
            
            # Compute loss
            loss = F.mse_loss(outputs[..., :3], batch_targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                logger.info(f"Iteration {i}, Loss: {loss.item():.4f}")
                
        # Save model
        torch.save(self.state_dict(), Path(output_dir) / "nerf_model.pth")

    def render(self, pose, height=None, width=None):
        """Render an image from a given pose"""
        self.eval()
        height = height or config.IMAGE_SIZE[0]
        width = width or config.IMAGE_SIZE[1]
        
        # Generate rays
        rays_o, rays_d, points, t_vals = self.ray_gen.generate_rays(pose)
        points = points.view(-1, 3)
        
        # Render in batches
        all_outputs = []
        for i in range(0, points.shape[0], config.BATCH_SIZE):
            with torch.no_grad():
                batch = points[i:i+config.BATCH_SIZE].to(self.device)
                outputs = self(batch)
                all_outputs.append(outputs.cpu())
                
        # Composite the final image
        rendered = torch.cat(all_outputs, dim=0)
        image = rendered[..., :3].view(height, width, 3).numpy()
        
        return (image * 255).astype(np.uint8)

if __name__ == "__main__":
    config.ensure_directories()
    
    # Example usage
    nerf = NeRFWrapper()
    
    # Load your dataset
    images = [...]  # List of torch tensors (H, W, 3)
    poses = [...]   # List of camera poses (3x4 or 4x4 matrices)
    
    # Train the model
    nerf.train(images, poses, config.MODELS_DIR, num_iters=5000)
    
    # Render a test view
    test_pose = [...]  # Test camera pose
    rendered = nerf.render(test_pose)
    cv2.imwrite(str(config.VISUALIZATIONS_DIR / "nerf_render.png"), 
               cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))