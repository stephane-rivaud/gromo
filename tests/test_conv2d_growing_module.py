from copy import deepcopy
from unittest import TestCase, main

import torch

from gromo.conv2d_growing_module import Conv2dGrowingModule
from gromo.utils.utils import global_device


class MyTestCase(TestCase):
    def setUp(self):
        self.demo_layer = torch.nn.Conv2d(
            2, 7, (3, 5), bias=False, device=global_device()
        )
        self.demo = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=(3, 5), use_bias=False
        )
        self.demo.layer = self.demo_layer

        self.demo_layer_b = torch.nn.Conv2d(
            2, 7, 3, padding=1, bias=True, device=global_device()
        )
        self.demo_b = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=3, padding=1, use_bias=True
        )
        self.demo_b.layer = self.demo_layer_b

        torch.manual_seed(0)
        self.input_x = torch.randn(5, 2, 10, 10, device=global_device())

    def test_init(self):
        # no bias
        m = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=(3, 5), use_bias=False
        )
        self.assertIsInstance(m, Conv2dGrowingModule)

        # with bias
        m = Conv2dGrowingModule(
            in_channels=2, out_channels=7, kernel_size=3, padding=1, use_bias=True
        )
        self.assertIsInstance(m, Conv2dGrowingModule)
        self.assertEqual(m.layer.padding, (1, 1))
        self.assertTrue(m.layer.bias is not None)
        self.assertEqual(m.layer.kernel_size, (3, 3))

    def test_forward(self):
        # no bias
        y = self.demo(self.input_x)
        self.assertTrue(torch.equal(y, self.demo_layer(self.input_x)))

        # with bias
        y = self.demo_b(self.input_x)
        self.assertTrue(torch.equal(y, self.demo_layer_b(self.input_x)))

    def test_number_of_parameters(self):
        # no bias
        self.assertEqual(self.demo.number_of_parameters(), self.demo_layer.weight.numel())

        # with bias
        self.assertEqual(
            self.demo_b.number_of_parameters(),
            self.demo_layer_b.weight.numel() + self.demo_layer_b.bias.numel(),
        )

    def test_str(self):
        self.assertIsInstance(str(self.demo), str)
        self.assertIsInstance(repr(self.demo), str)
        self.assertIsInstance(str(self.demo_b), str)
        for i in (0, 1, 2):
            self.assertIsInstance(self.demo.__str__(i), str)

    def test_layer_of_tensor(self):
        # no bias
        wl = self.demo.layer_of_tensor(self.demo_layer.weight.data)
        # way to test that wl == self.demo_layer
        y = self.demo(self.input_x)
        self.assertTrue(torch.equal(y, wl(self.input_x)))

        with self.assertRaises(AssertionError):
            _ = self.demo.layer_of_tensor(
                self.demo_layer.weight.data, self.demo_layer_b.bias.data
            )

        # with bias
        wl = self.demo_b.layer_of_tensor(
            self.demo_layer_b.weight.data, self.demo_layer_b.bias.data
        )
        # way to test that wl == self.demo_layer
        y = self.demo_b(self.input_x)
        self.assertTrue(torch.equal(y, wl(self.input_x)))

        with self.assertRaises(AssertionError):
            _ = self.demo_b.layer_of_tensor(self.demo_layer_b.weight.data)

    def test_layer_in_extension(self):
        in_extension = torch.nn.Conv2d(3, 7, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        local_demo.layer_in_extension(in_extension.weight)

        torch.manual_seed(0)
        x = torch.randn(23, 5, 10, 10, device=global_device())
        x_main = x[:, :2]
        x_ext = x[:, 2:]
        y_th = self.demo(x_main) + in_extension(x_ext)
        y = local_demo(x)
        self.assertTrue(
            torch.allclose(y, y_th, atol=1e-6),
            f"Error: ({torch.abs(y - y_th).max().item():.2e})",
        )

    def test_layer_out_extension_without_bias(self):
        out_extension = torch.nn.Conv2d(2, 5, (3, 5), bias=False, device=global_device())
        local_demo = deepcopy(self.demo)
        with self.assertWarns(UserWarning):
            local_demo.layer_out_extension(
                out_extension.weight, torch.empty(out_extension.out_channels)
            )

        y_main_th = self.demo(self.input_x)
        y_ext_th = out_extension(self.input_x)
        y = local_demo(self.input_x)
        y_main = y[:, : self.demo.out_channels]
        y_ext = y[:, self.demo.out_channels :]
        self.assertTrue(
            torch.allclose(y_main, y_main_th, atol=1e-6),
            f"Error: ({torch.abs(y_main - y_main_th).max().item():.2e})",
        )
        self.assertTrue(
            torch.allclose(y_ext, y_ext_th, atol=1e-6),
            f"Error: ({torch.abs(y_ext - y_ext_th).max().item():.2e})",
        )

    def test_layer_out_extension_with_bias(self):
        out_extension = torch.nn.Conv2d(
            2, 5, 3, bias=True, device=global_device(), padding=1
        )
        local_demo = deepcopy(self.demo_b)
        local_demo.layer_out_extension(out_extension.weight, out_extension.bias)

        y_main_th = self.demo_b(self.input_x)
        y_ext_th = out_extension(self.input_x)
        y = local_demo(self.input_x)
        y_main = y[:, : self.demo_b.out_channels]
        y_ext = y[:, self.demo_b.out_channels :]
        self.assertTrue(
            torch.allclose(y_main, y_main_th, atol=1e-6),
            f"Error: ({torch.abs(y_main - y_main_th).max().item():.2e})",
        )
        self.assertTrue(
            torch.allclose(y_ext, y_ext_th, atol=1e-6),
            f"Error: ({torch.abs(y_ext - y_ext_th).max().item():.2e})",
        )

    def test_tensor_s_update_without_bias(self):
        self.demo.init_computation()
        self.demo(self.input_x)

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, self.input_x.size(0))

        self.demo(self.input_x)
        self.demo.tensor_s.update()
        self.assertEqual(self.demo.tensor_s.samples, 2 * self.input_x.size(0))

        f = self.demo.in_channels * self.demo.kernel_size[0] * self.demo.kernel_size[1]
        self.assertEqual(self.demo.tensor_s().shape, (f, f))
        self.assertTrue(
            torch.allclose(self.demo.tensor_s(), self.demo.tensor_s().transpose(0, 1))
        )

        self.demo.reset_computation()

    def test_tensor_s_update_with_bias(self):
        self.demo_b.init_computation()
        self.demo_b(self.input_x)

        self.demo_b.tensor_s.update()
        self.assertEqual(self.demo_b.tensor_s.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        f = (
            self.demo_b.in_channels
            * self.demo_b.kernel_size[0]
            * self.demo_b.kernel_size[1]
            + 1
        )
        self.assertEqual(self.demo_b.tensor_s().shape, (f, f))
        self.assertEqual(
            self.demo_b.tensor_s()[-1, -1], self.input_x.size(2) * self.input_x.size(3)
        )
        # we do the average on the number of samples n but
        # should we not do it on the number of blocks n * h * w ?
        self.assertTrue(
            torch.allclose(self.demo_b.tensor_s(), self.demo_b.tensor_s().transpose(0, 1))
        )

        self.demo.reset_computation()

    def test_tensor_m_update_without_bias(self):
        self.demo.init_computation()
        y = self.demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, self.input_x.size(0))

        y = self.demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()
        self.demo.tensor_m.update()
        self.assertEqual(self.demo.tensor_m.samples, 2 * self.input_x.size(0))

        f = self.demo.in_channels * self.demo.kernel_size[0] * self.demo.kernel_size[1]
        self.assertEqual(self.demo.tensor_m().shape, (f, self.demo.out_channels))
        self.demo.reset_computation()

    def test_tensor_m_update_with_bias(self):
        self.demo_b.init_computation()
        y = self.demo_b(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo_b.tensor_m.update()
        self.assertEqual(self.demo_b.tensor_m.samples, self.input_x.size(0))
        # TODO: improve the specificity of the test
        # here we only test that the number of samples is correct

        f = (
            self.demo_b.in_channels
            * self.demo_b.kernel_size[0]
            * self.demo_b.kernel_size[1]
            + 1
        )
        self.assertEqual(self.demo_b.tensor_m().shape, (f, self.demo_b.out_channels))
        self.demo.reset_computation()

    def test_compute_optimal_delta_without_bias(self):
        self.demo.init_computation()
        y = self.demo(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo.tensor_s.update()
        self.demo.tensor_m.update()

        self.demo.compute_optimal_delta()
        self.assertEqual(
            self.demo.delta_raw.shape,
            (
                self.demo.out_channels,
                self.demo.in_channels
                * self.demo.kernel_size[0]
                * self.demo.kernel_size[1],
            ),
        )
        self.assertTrue(self.demo.optimal_delta_layer is not None)
        self.assertIsInstance(self.demo.optimal_delta_layer, torch.nn.Conv2d)
        # TODO: improve the specificity of the test

        self.demo.compute_optimal_delta(dtype=torch.float64)
        self.assertIsInstance(self.demo.optimal_delta_layer, torch.nn.Conv2d)

        self.demo.reset_computation()
        self.demo.delete_update()

    def test_compute_optimal_delta_with_bias(self):
        self.demo_b.init_computation()
        y = self.demo_b(self.input_x)
        loss = torch.norm(y)
        loss.backward()

        self.demo_b.tensor_s.update()
        self.demo_b.tensor_m.update()

        self.demo_b.compute_optimal_delta()
        self.assertEqual(
            self.demo_b.delta_raw.shape,
            (
                self.demo_b.out_channels,
                self.demo_b.in_channels
                * self.demo_b.kernel_size[0]
                * self.demo_b.kernel_size[1]
                + 1,
            ),
        )
        self.assertTrue(self.demo_b.optimal_delta_layer is not None)
        self.assertIsInstance(self.demo_b.optimal_delta_layer, torch.nn.Conv2d)
        self.assertTrue(self.demo_b.optimal_delta_layer.bias is not None)
        # TODO: improve the specificity of the test

        self.demo_b.reset_computation()
        self.demo_b.delete_update()

    def test_mask_tensor_t(self):
        with self.assertRaises(AssertionError):
            _ = self.demo.mask_tensor_t

        hin, win = 11, 13
        x = torch.randn(1, 2, hin, win, device=global_device())
        hout, wout = self.demo(x).shape[2:]
        self.demo.input_size = (hin, win)

        tensor_t = self.demo.mask_tensor_t

        self.assertIsInstance(tensor_t, torch.Tensor)
        self.assertIsInstance(self.demo._mask_tensor_t, torch.Tensor)

        size_theoretic = (
            hout * wout,
            self.demo.kernel_size[0] * self.demo.kernel_size[1],
            hin * win,
        )
        for i, (t, t_th) in enumerate(zip(tensor_t.shape, size_theoretic)):
            self.assertEqual(t, t_th, f"Error for dim {i}: should be {t_th}, got {t}")

    # test input extended
    # test compute m prev update : how ?
    # test cross covariance update : how ?
    # test compute compute_prev_s_update update : how ?


if __name__ == "__main__":
    main()
