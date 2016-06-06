require 'nn'

local Zero, parent = torch.class('nn.Zero', 'nn.Module')

function Zero:__init(nInputDim, reduce)
  parent.__init(self)
  self.nInputDim = nInputDim
  self.reduce = reduce or true
end

function Zero:updateOutput(input)
  if self.reduce then
    if input:dim() == self.nInputDim then
      self.output:resize(1)
    else
      self.output:resize(input:size(1), 1)
    end
  else
    self.output:resizeAs(input)
  end
  self.output:zero()
  return self.output
end

function Zero:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  return self.gradInput
end
