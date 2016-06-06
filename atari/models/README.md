# models  

This is where to place the models. All models follow a standardized structure:

* `model`
  * `build(struct)`: builds the actual `nn` network based on the architecture specified in struct
  * `parallelBuild(network)`: pushes the `nn` network into a `DataParallelTable` for data parallelism in a multi-gpu environment.
  * `getNetwork()`: returns the `nn` network inside the model class.
  * `getParameters()`: gets the flattened parameters and gradients from the `nn` network.
  * `feval(x, minibatch)`: evaluates the loss function using parameter weights `x` for the specified `minibatch`.
  * `sendRecord()`: returns a record of the loss function value(s) at the current iteration (used in combination with `Logger`. See `utils/Logger`).
  * `cuda()`: converts `nn` network to GPU-based
