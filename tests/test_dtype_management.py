import pytest
import torch
import torch.nn as nn
from typing import Type, Any, Dict, Tuple, List, Optional

from gromo.containers.growing_mlp import GrowingMLP
from gromo.containers.growing_residual_mlp import GrowingResidualMLP
from gromo.containers.growing_mlp_mixer import GrowingMLPMixer
from gromo.modules.linear_growing_module import LinearGrowingModule
from gromo.modules.conv2d_growing_module import Conv2dGrowingModule
from tests.torch_unittest import TorchTestCase
from tests.unittest_tools import unittest_parametrize
from gromo.utils.utils import reset_device, reset_dtype


def get_available_dtypes() -> List[Dict[str, Optional[torch.dtype]]]:
    """Get list of available dtypes for testing."""
    dtypes = [
        {"dtype": torch.float16},
        # {"dtype": torch.bfloat16},
        {"dtype": torch.float32},
        {"dtype": torch.float64},
    ]
    return dtypes

# Get available dtypes once at module level
AVAILABLE_DTYPES = get_available_dtypes()


class TestDtypeManagement(TorchTestCase):
    """Test dtype management for growing modules."""
    
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

        # Reset device and dtype to ensure that the tests are not affected by previous tests.
        reset_device()
        reset_dtype()
        
    def _test_module_operations(
        self, 
        module_class: Type[nn.Module], 
        input_shape: Tuple[int, ...],
        dtype: torch.dtype | None,
        **module_kwargs: Dict[str, Any]
    ) -> None:
        """Test operations on a module with the given dtype."""
        # Create module and set dtype for parameters
        module = module_class(**module_kwargs)
        
        # Convert module parameters to the target dtype
        module.to(dtype=dtype)
        
        # Verify all parameters are in the correct dtype
        for param in module.parameters():
            self.assertEqual(param.dtype, dtype, 
                          f"Parameter should be {dtype}, but is {param.dtype}")
        
        # Initialize computation
        module.init_computation()
        
        # Create test input with the same dtype
        x = torch.randn((self.batch_size, *input_shape), 
                       dtype=dtype, 
                       requires_grad=True)
        
        # Test forward pass
        out = module(x)
        self.assertEqual(out.dtype, dtype, 
                        f"Output should be {dtype}, but is {out.dtype}")
        
        # Skip gradient checks for half-precision dtypes due to potential numerical instability
        # if dtype in (torch.float32, torch.float64):
        # Test backward pass
        loss = out.sum()
        loss.backward()
        
        # Check gradients
        for param in module.parameters():
            if param.grad is not None:  # Some parameters might not have gradients
                self.assertEqual(param.grad.dtype, dtype,
                                f"Gradient should be {dtype}, but is {param.grad.dtype}")
                self.assertFalse(torch.isnan(param.grad).any(), 
                                f"Gradient contains NaN values for dtype {dtype}")
        
        # Update computation
        module.update_computation()

        # Verify that statistics are in the correct dtype
        if module_class == LinearGrowingModule or module_class == Conv2dGrowingModule:
            statistics_name = ['tensor_s', 'tensor_m', 'tensor_m_prev', 'cross_covariance']
            if hasattr(module, 's_growth_is_needed') and module.s_growth_is_needed:
                statistics_name.append('tensor_s_growth')
            for stat_name in statistics_name:
                self.assertEqual(getattr(module, stat_name).dtype, dtype,
                                f"Statistic {stat_name} should be {dtype}, but is {getattr(module, stat_name).dtype}")
        
        # Test compute_optimal_update (if available)
        module.compute_optimal_updates()
        
        # Test extended_forward (if available)
        if hasattr(module, 'scaling_factor') and hasattr(module, 'extended_forward'):
            module.scaling_factor.data = torch.tensor(1.0, dtype=dtype)
            module.extended_forward(x)
    
    @unittest_parametrize(AVAILABLE_DTYPES)
    def test_growing_mlp(self, dtype):
        """Test GrowingMLP with different dtypes."""
        self._test_module_operations(
            GrowingMLP,
            input_shape=(self.in_features,),
            dtype=dtype,
            in_features=self.in_features,
            out_features=self.out_features,
            hidden_size=self.hidden_size,
            number_hidden_layers=2,
        )
    
    @unittest_parametrize(AVAILABLE_DTYPES)
    def test_growing_residual_mlp(self, dtype):
        """Test GrowingResidualMLP with different dtypes."""
        self._test_module_operations(
            GrowingResidualMLP,
            input_shape=(self.in_features,),
            dtype=dtype,
            in_features=self.in_features,
            out_features=self.out_features,
            num_features=self.hidden_size,
            hidden_features=self.hidden_size,
            num_blocks=2,
        )
    
    @unittest_parametrize(AVAILABLE_DTYPES)
    def test_growing_mlp_mixer(self, dtype):
        """Test GrowingMLPMixer with different dtypes."""
        self._test_module_operations(
            GrowingMLPMixer,
            input_shape=(3, 32, 32),  # CIFAR-like input
            dtype=dtype,
            in_features=(3, 32, 32),
            out_features=10,
            patch_size=4,
            num_features=64,
            hidden_dim_token=32,
            hidden_dim_channel=128,
            num_blocks=2,
        )
    
    @unittest_parametrize(AVAILABLE_DTYPES)
    def test_linear_growing_module(self, dtype):
        """Test LinearGrowingModule with different dtypes."""
        self._test_module_operations(
            LinearGrowingModule,
            input_shape=(self.in_features,),
            dtype=dtype,
            in_features=self.in_features,
            out_features=self.out_features,
            post_layer_function=nn.ReLU(),
        )
    
    @unittest_parametrize(AVAILABLE_DTYPES)
    def test_conv2d_growing_module(self, dtype):
        """Test Conv2dGrowingModule with different dtypes."""
        self._test_module_operations(
            Conv2dGrowingModule,
            input_shape=(3, 32, 32),
            dtype=dtype,
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
            input_size=(32, 32),
        )

    def tearDown(self):
        reset_device()
        reset_dtype()
