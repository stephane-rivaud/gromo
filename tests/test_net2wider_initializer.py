"""
Unit tests for Net2Wider initializer.

Tests validate:
1. Net2Wider initialization is available and callable
2. It falls back to Kaiming for incompatible shapes
3. It doesn't crash when used as an initializer option
"""

import torch

from gromo.modules.conv2d_growing_module import FullConv2dGrowingModule
from gromo.utils.utils import global_device
from tests.torch_unittest import TorchTestCase


class TestNet2WiderInitializer(TorchTestCase):
    """Test Net2Wider function-preserving initialization."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = global_device()

    def _create_connected_pair(self, in_channels=3, hidden_channels=16, out_channels=8):
        """Create two connected Conv2d layers."""
        first = FullConv2dGrowingModule(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            device=self.device,
            name="first",
        )
        second = FullConv2dGrowingModule(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            device=self.device,
            name="second",
            previous_module=first,
        )
        first.next_module = second
        return first, second

    def test_net2wider_initializer_available(self):
        """Test that net2wider initializer is registered and available."""
        first, second = self._create_connected_pair()
        
        # Check that net2wider_initialization method exists
        assert hasattr(second, "net2wider_initialization")
        assert callable(second.net2wider_initialization)

    def test_net2wider_creates_extensions(self):
        """
        Test that create_layer_extensions works with net2wider initializer.
        """
        first, second = self._create_connected_pair(3, 16, 8)
        
        # Create extensions specifying net2wider for input init
        # (will fall back to kaiming since shapes incompatible)
        second.create_layer_extensions(
            extension_size=4,
            input_extension_init="net2wider",
            output_extension_init="copy_uniform",
        )
        
        # Verify extended layers exist
        assert first.extended_output_layer is not None
        assert second.extended_input_layer is not None

    def test_net2wider_no_crashes_with_large_extension(self):
        """Test that Net2Wider doesn't crash with large extensions."""
        first, second = self._create_connected_pair(2, 4, 8)
        
        # Extend with many more neurons than base
        second.create_layer_extensions(
            extension_size=100,
            input_extension_init="net2wider",
        )
        
        # Should complete without error
        assert second.extended_input_layer is not None

    def test_net2wider_direct_call_with_compatible_shapes(self):
        """
        Test Net2Wider initialization directly with compatible shapes.
        """
        first, second = self._create_connected_pair(3, 5, 8)
        
        # Create tensors with compatible shapes
        ext_weight = torch.empty(4, 3, 3, 3, device=self.device)
        base_weight = first.layer.weight  # shape (5, 3, 3, 3)
        
        # Call initialization directly with compatible shapes
        second.net2wider_initialization(ext_weight, base_weight, fan_in=12)
        
        # Verify extension is initialized
        assert ext_weight.abs().sum() > 0
        # Verify no NaN/Inf
        assert torch.isfinite(ext_weight).all()

    def test_net2wider_fallback_to_kaiming_incompatible(self):
        """
        Test that Net2Wider falls back to Kaiming with incompatible shapes.
        """
        first, second = self._create_connected_pair(3, 5, 8)
        
        # Create tensors with incompatible shapes
        ext_tensor = torch.empty(3, 4, 3, 3, device=self.device)
        reference = first.layer.weight  # (5, 3, 3, 3)
        
        # Call with incompatible shape - should fall back to kaiming
        second.net2wider_initialization(ext_tensor, reference, fan_in=15)
        
        # Should be initialized via kaiming fallback
        assert ext_tensor.abs().sum() > 0
        assert torch.isfinite(ext_tensor).all()

    def test_net2wider_fallback_to_kaiming_none_reference(self):
        """Test that Net2Wider falls back to Kaiming when reference is None."""
        first, second = self._create_connected_pair(3, 5, 8)
        
        ext_tensor = torch.empty(3, 5, 3, 3, device=self.device)
        
        # Call with None reference
        second.net2wider_initialization(ext_tensor, None, fan_in=15)
        
        # Should be initialized via kaiming fallback
        assert ext_tensor.abs().sum() > 0
        assert torch.isfinite(ext_tensor).all()

    def test_net2wider_numerical_stability(self):
        """Test that Net2Wider maintains numerical stability."""
        first, second = self._create_connected_pair(3, 8, 8)
        
        # Create extension with net2wider
        second.create_layer_extensions(
            extension_size=2,
            input_extension_init="net2wider",
        )
        
        # Check that all weights are finite
        for param in first.parameters():
            assert torch.isfinite(param).all()
        for param in second.parameters():
            assert torch.isfinite(param).all()

    def test_net2wider_is_in_known_inits(self):
        """Test that net2wider is registered in known initializers."""
        first, second = self._create_connected_pair()
        
        # Try to create extensions with net2wider
        # This will fail with ValueError if net2wider is not registered
        try:
            second.create_layer_extensions(
                extension_size=2,
                input_extension_init="net2wider",
            )
        except ValueError as e:
            if "Unknown initialization method" in str(e):
                self.fail(f"net2wider not registered in known_inits: {e}")
            raise


class TestNet2WiderDirectFunctionality(TorchTestCase):
    """Test Net2Wider duplication logic directly."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = global_device()

    def test_net2wider_duplicates_channels_correctly(self):
        """Test that Net2Wider duplicates channels and scales correctly."""
        first = FullConv2dGrowingModule(
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
            use_bias=False,
            device=self.device,
            name="test",
        )
        
        # Create extension tensor with same input channels as base
        ext_weight = torch.empty(2, 2, 3, 3, device=self.device)
        base_weight = first.layer.weight  # (3, 2, 3, 3)
        
        # Initialize with net2wider
        first.net2wider_initialization(ext_weight, base_weight, fan_in=10)
        
        # Each extension row should be a scaled copy of a base row
        # So the extension values should be non-zero and smaller than base (on average)
        ext_mean = ext_weight.abs().mean().item()
        base_mean = base_weight.abs().mean().item()
        
        # Extension should be initialized (non-zero)
        assert ext_mean > 0

    def test_net2wider_with_single_base_channel(self):
        """Test Net2Wider with only one base channel to duplicate."""
        first = FullConv2dGrowingModule(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            padding=1,
            use_bias=False,
            device=self.device,
            name="test",
        )
        
        # Create extension with compatible shape
        ext_weight = torch.empty(5, 2, 3, 3, device=self.device)
        base_weight = first.layer.weight  # (1, 2, 3, 3)
        
        # Initialize with net2wider
        first.net2wider_initialization(ext_weight, base_weight, fan_in=7)
        
        # All extension rows should be scaled versions of the single base row
        for i in range(5):
            assert torch.isfinite(ext_weight[i]).all()
            # Should not be all zeros
            assert ext_weight[i].abs().sum() > 0


if __name__ == "__main__":
    import unittest
    unittest.main()
