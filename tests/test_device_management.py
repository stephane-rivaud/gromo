import pytest
import torch
import torch.nn as nn
from typing import Type, Any, Dict, Tuple, List

from gromo.containers.growing_mlp import GrowingMLP
from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.containers.growing_mlp_mixer import GrowingMLPMixer
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize


def get_available_devices() -> List[Dict[str, str]]:
    """Get list of available devices for testing."""
    devices = [{"device": "cpu"}]
    if torch.backends.mps.is_available():
        devices.append({"device": "mps"})
    if torch.cuda.is_available():
        devices.append({"device": "cuda"})
    return devices

# Get available devices once at module level
AVAILABLE_DEVICES = get_available_devices()


class TestDeviceManagement(TorchTestCase):
    """Test device management for growing modules."""
    
    # Test parameters
    batch_size = 2
    in_features = 10
    out_features = 5
    hidden_size = 8
    num_blocks = 2
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
    def _test_module_operations(
        self, 
        module_class: Type[nn.Module], 
        input_shape: Tuple[int, ...],
        device: str,
        **module_kwargs: Dict[str, Any]
    ) -> None:
        """Test operations on a module with the given device."""
        device_torch = torch.device(device)
        
        # Create module and move to device
        module = module_class(**module_kwargs)
        module.to(device_torch)

        # init computation
        module.init_computation()
        
        # Create test input
        x = torch.randn((self.batch_size, *input_shape), device=device_torch, requires_grad=True)
        
        # Test forward pass
        out = module(x)
        self.assertEqual(out.device.type, x.device.type, f"Device should be {device_torch.type}, but is {out.device.type}")
        
        # Test backward pass
        out = module(x)
        loss = out.sum()
        loss.backward()
        # Check gradients
        for param in module.parameters():
            self.assertEqual(param.grad.device.type, device_torch.type, \
                f"Gradient device should be {device_torch.type}, but is {param.grad.device.type}")
            self.assertFalse(torch.isnan(param.grad).any(), "Gradient contains NaN values")
        
        # Update computation
        module.update_computation()
        
        # Test compute_optimal_update (if available)
        if hasattr(module, 'compute_optimal_updates'):
            module.compute_optimal_updates()
        
        # Test extended_forward (if available)
        if hasattr(module, 'scaling_factor') and hasattr(module, 'extended_forward'):
            module.scaling_factor.data = torch.tensor(1.0, device=device_torch)
            module.extended_forward(x)
    
    @unittest_parametrize(AVAILABLE_DEVICES)
    def test_growing_mlp(self, device):
        """Test GrowingMLP on all available devices."""
        self._test_module_operations(
            GrowingMLP,
            input_shape=(self.in_features,),
            device=device,
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=2,
        )
    
    @unittest_parametrize(AVAILABLE_DEVICES)
    def test_growing_residual_mlp(self, device):
        """Test GrowingResidualMLP on all available devices."""
        self._test_module_operations(
            GrowingResidualMLP,
            input_shape=(self.in_features,),
            device=device,
            in_features=self.in_features,
            out_features=self.out_features,
            num_features=self.hidden_size,
            hidden_features=self.hidden_size,
            num_blocks=2,
        )
    
    @unittest_parametrize(AVAILABLE_DEVICES)
    def test_growing_mlp_mixer(self, device):
        """Test GrowingMLPMixer on all available devices."""
        self._test_module_operations(
            GrowingMLPMixer,
            input_shape=(3, 32, 32),  # CIFAR-like input
            device=device,
            in_features=(3, 32, 32),
            out_features=10,
            patch_size=4,
            num_features=64,
            hidden_dim_token=32,
            hidden_dim_channel=128,
            num_blocks=2,
        )
    
    @unittest_parametrize(AVAILABLE_DEVICES)
    def test_linear_growing_module(self, device):
        """Test LinearGrowingModule on all available devices."""
        self._test_module_operations(
            LinearGrowingModule,
            input_shape=(self.in_features,),
            device=device,
            in_features=self.in_features,
            out_features=self.out_features,
            post_layer_function=nn.ReLU(),
        )
    
    @unittest_parametrize(AVAILABLE_DEVICES)
    def test_conv2d_growing_module(self, device):
        """Test Conv2dGrowingModule on all available devices."""
        self._test_module_operations(
            Conv2dGrowingModule,
            input_shape=(3, 32, 32),
            device=device,
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            input_size=(32, 32),
        )
