# NN Library Utilities #
Simple Modules are used for various tasks like adapting Tensor methods and providing affine transformations :

  * Parameterized Modules :
    * [Linear](#nn.Linear) : a linear transformation ;
    * [SparseLinear](#nn.SparseLinear) : a linear transformation with sparse inputs ;
    * [Bilinear](#nn.Bilinear) : a bilinear transformation with sparse inputs ;
    * [PartialLinear](#nn.PartialLinear) : a linear transformation with sparse inputs with the option of only computing a subset ;
    * [Add](#nn.Add) : adds a bias term to the incoming data ;
    * [Mul](#nn.Mul) : multiply a single scalar factor to the incoming data ;
    * [CMul](#nn.CMul) : a component-wise multiplication to the incoming data ;
    * [Euclidean](#nn.Euclidean) : the euclidean distance of the input to `k` mean centers ;
    * [WeightedEuclidean](#nn.WeightedEuclidean) : similar to [Euclidean](#nn.Euclidean), but additionally learns a diagonal covariance matrix ;
    * [Cosine](#nn.Cosine) : the cosine similarity of the input to `k` mean centers ;
  * Modules that adapt basic Tensor methods :
    * [Copy](#nn.Copy) : a [copy](https://github.com/torch/torch7/blob/master/doc/tensor.md#torch.Tensor.copy) of the input with [type](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-or-string-typetype) casting ;
    * [Narrow](#nn.Narrow) : a [narrow](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-narrowdim-index-size) operation over a given dimension ;
    * [Replicate](#nn.Replicate) : [repeats](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-repeattensorresult-sizes) input `n` times along its first dimension ;
    * [Reshape](#nn.Reshape) : a [reshape](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchreshaperes-x-m-n) of the inputs ;
    * [View](#nn.View) : a [view](https://github.com/torch/torch7/blob/master/doc/tensor.md#result-viewresult-tensor-sizes) of the inputs ;
    * [Contiguous](#nn.Contiguous) : [contiguous](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-contiguous) of the inputs ;
    * [Select](#nn.Select) : a [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index) over a given dimension ;
    * [MaskedSelect](#nn.MaskedSelect) : a [masked select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-maskedselect-index) module performs the torch.maskedSelect operation ;
    * [Index](#nn.Index) : a [index](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-indexdim-index) over a given dimension ;
    * [Squeeze](#nn.Squeeze) : [squeezes](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-squeezedim) the input;
    * [Unsqueeze](#nn.Unsqueeze) : unsqueeze the input, i.e., insert singleton dimension;
    * [Transpose](#nn.Transpose) : [transposes](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-transposedim1-dim2) the input ;
  * Modules that adapt mathematical Tensor methods :
    * [AddConstant](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.AddConstant) : adding a constant ;
    * [MulConstant](https://github.com/torch/nn/blob/master/doc/transfer.md#nn.MulConstant) : multiplying a constant ;
    * [Max](#nn.Max) : a [max](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.max) operation over a given dimension ;
    * [Min](#nn.Min) : a [min](https://github.com/torch/torch7/blob/master/doc/maths.md#torchminresval-resind-x) operation over a given dimension ;
    * [Mean](#nn.Mean) : a [mean](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchmeanres-x-dim) operation over a given dimension ;
    * [Sum](#nn.Sum) : a [sum](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchsumres-x) operation over a given dimension ;
    * [Exp](#nn.Exp) : an element-wise [exp](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchexpres-x) operation ;
    * [Log](#nn.Log) : an element-wise [log](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchlogres-x) operation ;
    * [Abs](#nn.Abs) : an element-wise [abs](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchabsres-x) operation ;
    * [Power](#nn.Power) : an element-wise [pow](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchpowres-x) operation ;
    * [Square](#nn.Square) : an element-wise square operation ;
    * [Sqrt](#nn.Sqrt) : an element-wise [sqrt](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchsqrtres-x) operation ;
    * [Clamp](#nn.Clamp) : an element-wise [clamp](https://github.com/torch/torch7/blob/master/doc/maths.md#res-torchclampres-tensor1-min_value-max_value) operation ;
    * [Normalize](#nn.Normalize) : normalizes the input to have unit `L_p` norm ;
    * [MM](#nn.MM) : matrix-matrix multiplication (also supports batches of matrices) ;
  * Miscellaneous Modules :
    * [BatchNormalization](#nn.BatchNormalization) : mean/std normalization over the mini-batch inputs (with an optional affine transform) ;
    * [Identity](#nn.Identity) : forward input as-is to output (useful with [ParallelTable](table.md#nn.ParallelTable)) ;
    * [Dropout](#nn.Dropout) : masks parts of the `input` using binary samples from a [bernoulli](http://en.wikipedia.org/wiki/Bernoulli_distribution) distribution ;
    * [SpatialDropout](#nn.SpatialDropout) : same as Dropout but for spatial inputs where adjacent pixels are strongly correlated ;
    * [VolumetricDropout](#nn.VolumetricDropout) : same as Dropout but for volumetric inputs where adjacent voxels are strongly correlated ;
    * [Padding](#nn.Padding) : adds padding to a dimension ;
    * [L1Penalty](#nn.L1Penalty) : adds an L1 penalty to an input (for sparsity) ;
    * [GradientReversal](#nn.GradientReversal) : reverses the gradient (to maximize an objective function) ;