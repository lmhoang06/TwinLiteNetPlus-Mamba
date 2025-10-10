#!/usr/bin/env python3
"""
Example script demonstrating how to use the TwinLiteNetPlus layer visualizer
"""

import torch
import numpy as np
from visualize_layer import ModelVisualizer, load_sample_image, create_model
import matplotlib.pyplot as plt

def example_usage():
    """Demonstrate basic usage of the visualizer"""
    
    # Example 1: Basic visualization with random weights
    print("=== Example 1: Basic Visualization ===")
    
    # Create model first
    model = create_model(
        config='nano',  # Use nano configuration
        model_path='',  # Empty path will use random weights
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize visualizer with the model
    visualizer = ModelVisualizer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create a sample input
    input_tensor = load_sample_image(size=(224, 224))
    input_tensor = input_tensor.to(visualizer.device)
    
    # Visualize feature heatmaps for encoder layers
    print("Visualizing feature heatmaps...")
    visualizer.visualize_feature_heatmaps(
        input_tensor=input_tensor,
        layer_name='encoder.level1',
        num_channels=8,
        save_path='./example_heatmaps.png'
    )
    
    # Visualize attention maps
    print("Visualizing attention maps...")
    visualizer.visualize_attention_maps(
        input_tensor=input_tensor,
        save_path='./example_attention.png'
    )
    
    # Visualize layer statistics
    print("Visualizing layer statistics...")
    visualizer.visualize_layer_statistics(
        input_tensor=input_tensor,
        save_path='./example_statistics.png'
    )
    
    # Cleanup
    visualizer.cleanup()
    
    print("Example 1 completed! Check the generated PNG files.")

def example_with_pretrained_model():
    """Example using a pretrained model (if available)"""
    
    print("\n=== Example 2: With Pretrained Model ===")
    
    # Path to your pretrained model (update this path)
    model_path = './pretrained/nano.pth'  # Update this path
    
    try:
        # Create model with pretrained weights
        model = create_model(
            config='nano',
            model_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize visualizer with the model
        visualizer = ModelVisualizer(
            model=model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create input
        input_tensor = load_sample_image(size=(224, 224))
        input_tensor = input_tensor.to(visualizer.device)
        
        # Visualize Effective Receptive Field
        print("Computing Effective Receptive Field...")
        visualizer.visualize_effective_receptive_field(
            input_size=(224, 224),
            target_layer='encoder.level3_0',
            save_path='./example_erf.png'
        )
        
        # Visualize gradient flow
        print("Visualizing gradient flow...")
        visualizer.visualize_gradient_flow(
            input_tensor=input_tensor,
            target_class=0,  # Drivable area
            save_path='./example_gradients.png'
        )
        
        # Cleanup
        visualizer.cleanup()
        
        print("Example 2 completed!")
        
    except FileNotFoundError:
        print(f"Pretrained model not found at {model_path}")
        print("Please download the pretrained weights or update the path.")

def example_custom_image():
    """Example using a custom image"""
    
    print("\n=== Example 3: Custom Image ===")
    
    # Create a synthetic driving scene
    def create_driving_scene(size=(224, 224)):
        """Create a synthetic driving scene for visualization"""
        img = np.zeros((*size, 3), dtype=np.float32)
        
        # Sky (top half)
        img[:size[0]//2, :, 2] = 0.7  # Blue sky
        
        # Road (bottom half)
        img[size[0]//2:, :, :] = 0.3  # Gray road
        
        # Lane markings
        img[size[0]//2:, size[1]//4, :] = 1.0  # White line
        img[size[0]//2:, 3*size[1]//4, :] = 1.0  # White line
        
        # Add some noise for realism
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)
        
        return img
    
    # Create synthetic image
    synthetic_img = create_driving_scene()
    
    # Convert to tensor
    input_tensor = torch.from_numpy(synthetic_img).permute(2, 0, 1).unsqueeze(0)
    
    # Create model
    model = create_model(
        config='small',  # Use small configuration
        model_path='',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Initialize visualizer
    visualizer = ModelVisualizer(
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    input_tensor = input_tensor.to(visualizer.device)
    
    # Visualize different layers
    layers = ['encoder.level1', 'encoder.level2_0', 'caam']
    
    for layer in layers:
        print(f"Visualizing {layer}...")
        visualizer.visualize_feature_heatmaps(
            input_tensor=input_tensor,
            layer_name=layer,
            num_channels=12,
            save_path=f'./example_{layer.replace(".", "_")}.png'
        )
    
    # Cleanup
    visualizer.cleanup()
    
    print("Example 3 completed!")

def main():
    """Run all examples"""
    print("TwinLiteNetPlus Layer Visualizer Examples")
    print("=" * 50)
    
    # Run examples
    example_usage()
    example_with_pretrained_model()
    example_custom_image()
    
    print("\nAll examples completed!")
    print("Check the generated PNG files for visualizations.")

if __name__ == '__main__':
    main()
