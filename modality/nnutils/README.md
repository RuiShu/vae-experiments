# nn library utilities

Extensions of the torch nn library for VAE-related layers

* Simple Modules:
  * [SplitTensor](#SplitTensor): Splits a Tensor input into a table of subtensors
  * [Sampler](#Sampler): Takes {mean, log_variance} as input and samples from the Gaussian distribution
  * [Zero](#Zero): Takes input and zeros it
* Criterion Modules:
  * [GaussianCriterion](#GaussianCriterion): Computes the Gaussian log-likelihood of evidence given a Gaussian distribution.
  * [KLDCriterion](#KLDCriterion): Computes the KL-Divergence between two Gaussian distributions.
  * [GmmKLDCriterion](#GmmKLDCriterion): Computes the KL-Divergence between a Gaussian distribution and Gaussian mixture.

<a name="SplitTensor"></a>
## SplitTensor ##
```lua
module = nn.SplitTensor(dimension, nInputDims, [splitSize = 1])
```
Splits tensor along the specific dimension. The following are equivalent
```lua
nn.SplitTensor(dimension, nInputDims, splitSize):forward(x)
x:split(splitSize, dimension)
```
`nInputDims` should be specified to differentiate passing in batch v. non-batch data.

<a name="Sampler"></a>
## Sampler ##
```lua
module = nn.Sampler()
```
Takes two tensors as input: `{mean, log_variance}`, and performs element-wise Gaussian sampling from the corresponding `(mean,log_variance)` pair.
```lua
nn.Sampler():forward({mean, log_variance})
```

<a name="Zero"></a>
## Zero ##
```lua
module = nn.Zero(nInputDim, [reduce = true])
```
Copies the input tensor and zeros it completely. 
```lua
nn.Zero(nInputDim, reduce):forward(x)
```
If `reduce = false`, the output will have the same dimensions as the input. If `reduce = true`, the output will have size `batchSize x 1` or `1` depending on whether a batch is passed in.


<a name="GaussianCriterion"></a>
## GaussianCriterion ##
```lua
module = nn.GaussianCriterion(nInputDim, [sizeAverage = false])
```
Computes the log-likelihood of a sample `x` given a Gaussian distribution `p`. If `sizeAverage = true`, computes the average log-likelihood over a minibatch.
```lua
nn.GaussianCriterion(nInputDim, sizeAverage):forward({mean, log_variance}, data)
```

<a name="KLDCriterion"></a>
## KLDCriterion ##
```lua
module = nn.KLDCriterion(nInputDim, [sizeAverage = false])
```
Computes the KL-divergence between two Gaussian distribution. If `sizeAverage = true`, computes the average log-likelihood over a minibatch.
```lua
nn.KLDCriterion(nInputDim, sizeAverage):forward({mu1, log_var1}, {mu2, log_var2})
```
The order of the distribution reflects the order in the `KL(p1 || p2)`, where `p1` is a distribution parameterized by `{mu1, log_var1}`, and `p2` is a distribution parameterized by `{mu2, log_var2}`. 

<a name="GmmKLDCriterion"></a>
## GmmKLDCriterion ##
```lua
module = nn.GmmKLDCriterion()
```
The API is the same as [KLDCriterion](#KLDCriterion):
```lua
nn.GmmKLDCriterion(nInputDim, sizeAverage):forward({mu1, log_var1}, {mu2, log_var2})
```
However, if the first distribution has dimensions `batchSize x Dim`, the second distribution has dimensions `batchSize x nMixtures x Dim`. Currently only supports where the `{mu1, log_var1}` tensors have dimensions `batchSize x Dim` or `Dim`. Does not support `sizeAverage` yet.


