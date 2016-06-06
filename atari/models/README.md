# models  

This is where to place the models. All models follow a standardized structure:

* `model`
  * `build(struct)`: builds the actual `nn` network based on the architecture specified in struct
  * `feval(x, minibatch)`: evaluates the loss function using parameter weights `x` for the specified `minibatch`.
  * `sendRecord()`: returns a record of the loss function value(s) at the current iteration (used in combination with `Logger`. See `utils/Logger`).
  * `cuda()`: converts `nn` network to GPU-based
