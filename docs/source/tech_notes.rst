=====================
GrowingModule (GroMo)
=====================

This is the documentation of the FoGro method described in [TMLR24] Verbockhaven, M., Rudkiewicz, T., Chevallier, S., and Charpiat, G. (2024). Growing tiny networks: Spotting expressivity bottlenecks and fixing them optimally. TMLR..

We propose to build growing networks using `GroMo` inherited from `torch.nn.Module`. We have two type of modules, the standard `GrowingModule` and the `AdditionGrowingModule`. The first one is a connections that contains the parameters of the network. The second one aims to connect multiple `GrowingModule` together.

----------------
Attributes
----------------

To compute the special updates (*i.e.* the ones that are different from classical gradient descent) we need to store tensor statistics. Those are computed using the `TensorStatistics` class as they are average over possibly multiple machines batches.

Here is a list of statistics that can be useful:

- `S local`: the tensor :math`S_{-1}` for the natural gradient
- `M`: the tensor $M_{-1}$ for the natural gradient
- `S prev`: the tensor $S_{-2}$ to compute the new weights
- `M prev`: the tensor $M_{-2}$ to compute the new weights
- `cross covariance`: the cross covariance $P$ to compute the new weights

To compute those statistics we need to access some intermediate quantities:

- $a$: the pre-activation of the layer
- $\nabla_{a} \mathcal{L}$: the gradient of the loss with respect to the pre-activation
- $b_{-1}$: the activation of the previous layer
- $b_{-2}$: the activation of two layer before

We store those quantities in the `GroMo` module directly. Depending of the situation it can be stored in different places but it is transparent for the user as we surcharge `input` and `pre_activity` to access those quantities in the same way.

.. image:: images/gromo_links.png
    :width: 800px
    :align: center
    :height: 565px
    :alt: Where are quantities stored ?

When we link only `GrowingModule` modules together the quantities are stored in the `GrowingModule` module. When we link `GrowingModule` with `AdditionGrowingModule` the quantities are stored in the `AdditionGrowingModule`.  This is due to the fact that we use the `AdditionGrowingModule` when we connect multiple `GrowingModule` together and we do not want to store the same quantities multiple times. Hence we store them in the `AdditionGrowingModule` and we can access them from the `GrowingModule`. Note that we perform the activation after the addition in the `AdditionGrowingModule` so that we can access the pre-activity of the previous layer as the input of the next layer.

----------------
Glossary
----------------

- machine batch / statistical batch: often in ML we use batch of data to estimate quantities and to process data together to make computation faster. For example in stochastic gradient descent we estimate the gradient with on batch of example that is computed at the same time. In our case we can compute multiple examples at the same time and do it multiple times to compute a statistics. We refer to the first one to machine batch and to the second as statistical batch.
