local FixedIndex, parent = torch.class('nn.FixedIndex', 'nn.Module')

function FixedIndex:__init(dimension, index, nInputDim)
  parent.__init(self)
  self.dimension = dimension    -- assumes non-batch
  self.nInputDim = nInputDim
  self.index = index:long()
end

function FixedIndex:updateOutput(input)
  local diff = input:dim() - self.nInputDim
  self.output:index(input, self.dimension + diff, self.index)
  return self.output
end

function FixedIndex:updateGradInput(input, gradOutput)
  local diff = input:dim() - self.nInputDim
  self.gradInput:resizeAs(input):zero()
  self.gradInput:indexAdd(self.dimension + diff, self.index, gradOutput)
  return self.gradInput
end

function FixedIndex:type(type, tensorCache)
  parent.type(self, type, tensorCache)
  self.index = self.index:long()
  return self
end
