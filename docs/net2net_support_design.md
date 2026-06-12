# Net2Wider Initializer Integration Design

## Overview
This document describes the Net2Wider neuron initialization support added to gromo for NeurIPS 2026 growth experiments.

## Scientific Goal
Implement and validate the Net2Wider initialization strategy (duplicating neurons with function preservation) so we can compare growth initialization strategies in DAG-growth experiments. We are NOT attempting to reproduce original Net2Net training curves, only to validate the initializer's function-preservation property.

## Architecture

### Insertion Points
1. **GrowingModule level**: New initializer `"net2wider"` added to the `known_inits` dictionary in `GrowingModule.create_layer_extensions()`.
   - Implements the function-preserving duplication rule for convolutional layers.
   - Duplicates selected output filters and divides their post-layer weights by replication count.

2. **GrowingBlock level**: Special handling in `GrowingBlock.create_layer_extensions()` for residual hidden-channel growth.
   - When growing hidden channels inside a residual block:
     - Extend `first_layer.output` (Conv2d) and `first_layer.post_layer_function` (BatchNorm1d)
     - Extend matching `second_layer.input` slice
     - **Do NOT extend** `second_layer.output` (preserves block output shape)
     - **Do NOT touch** `downsample` (preserves skip path)
   - This preserves the block boundary shapes while widening the internal computation.

### Initialization Strategy
**Net2Wider (Function-Preserving Duplication)**:
- For each new output channel, select an existing channel uniformly at random.
- Duplicate its pre-activation filter.
- Divide the post-layer weights (and biases if present) by the number of duplications to preserve the function value.
- For BatchNorm parameters: duplicate running mean/variance; duplicate scale/shift; adjust as needed for statistical consistency.

**Random (Default Control)**:
- Existing `"copy_uniform"` and `"kaiming"` initializers provide random baselines.
- No function preservation claimed.

### Why Skip Connections Are Unchanged
- ResNet blocks operate on fixed boundary shapes: `in_channels` → `out_channels`.
- The `downsample` path (if present) already handles mismatches between input and output.
- Widening the skip would require growing both `downsample` and all upstream layers in the path.
- Instead, we grow only the **hidden** channels **inside** the block (first → second layer).
- This preserves the block's external interface and DAG structure.

## Implementation Details

### GrowingModule Additions
- **`net2wider_initialization()`**: New method that implements function-preserving duplication for generic layers.
  - Takes: `tensor` (extension weights), `reference_tensor` (base layer weights), `fan_in` (input dimension)
  - Duplicates randomly selected rows from the reference tensor
  - For each replication, divides extension weights by replication count

### GrowingBlock Additions
- **`create_layer_extensions()` override**: Already delegates to `second_layer.create_layer_extensions()`.
  - Future enhancement: Add explicit residual-block-level configuration to select hidden-channel vs output-channel growth.
  - Currently, if a `GrowingBlock` is asked to grow, it grows the second layer's output (standard behavior).
  - Net2Wider support is available at the layer level; block-level growth selector is a future extension if needed.

### Configuration
- Hydra config in `experimental_grow`:
  - `hydra_script/aux_train_and_grow.py`: Pass `initializer_type` (e.g., `"random"` or `"net2wider"`) to `create_layer_extensions()`.
  - Default: `"copy_uniform"` (no change to existing behavior).
  - Optional override: `"net2wider"` for function-preserving growth.

## What Is Implemented
1. ✅ Net2Wider initialization function in `GrowingModule`.
2. ✅ Integration into `known_inits` dictionary.
3. ✅ Unit tests for function preservation (output delta < 1e-4 in eval mode).
4. ✅ Tests for parameter count, shape, and skip-path invariance.
5. ✅ Config plumbing in `experimental_grow` to select initializer.

## What Is NOT Implemented
- **Specialized hidden-channel growth in GrowingBlock**: Currently, block growth applies to output channels (standard). Hidden-channel growth is available via layer-level APIs but not exposed as a first-class block operation. This can be added later if DAG schedules require it.
- **Guaranteed function preservation in training mode** with BN running statistics. Tests validate eval-mode preservation only. In training, BN statistics will drift and function is not strictly preserved.
- **Multi-layer function preservation chains**: Net2Wider only applies to a single extension layer. Chaining duplication across multiple layers is not implemented.
- **Original Net2Net reproduction**: We do not attempt to match the original Net2Net paper's training curves or experimental setup.

## Testing Strategy
1. **Unit tests** (`test_net2wider_initializer.py`):
   - Create a simple Conv2d layer with known weights.
   - Extend with Net2Wider initialization.
   - Verify output delta < 1e-4 in eval mode.
   - Verify parameter count increases.
   - Verify skip/downsample unchanged (for blocks).

2. **Integration tests**:
   - Build a small ResNet with `GrowingBlock`s.
   - Apply Net2Wider growth to the first layer.
   - Check output shape, loss computation, and no NaN.

3. **No GPU requirement**: Tests are CPU-compatible but skip CUDA tests if not available.

## References
- Net2Net paper: https://arxiv.org/abs/1511.05641
- gromo growth API: `GrowingModule.create_layer_extensions()`, `GrowingBlock` residual structure.
- Related probe code (not integrated): `experimental_grow/experiments/probes/net2net_paper_reproduction.py` (reference only).
