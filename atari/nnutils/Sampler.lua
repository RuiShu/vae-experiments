require 'nn'

local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init(nInputDims)
  parent.__init(self)
  self.nInputDims = nInputDims
  self.eps = torch.Tensor()
  self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function Sampler:updateOutput(input)
  self:_viewInput(input)
  self.eps:resizeAs(self.lv):randn(self.lv:size())
  self.output:resizeAs(self.lv):copy(self.lv)
  self.output:div(2):exp():cmul(self.eps):add(self.mu)
  return self.output
end

function Sampler:updateGradInput(input, gradOutput)
  self.dMu = self.gradInput[1]:resizeAs(self.mu)
  self.dLv = self.gradInput[2]:resizeAs(self.lv)
  self.dMu:copy(gradOutput)
  self.dLv:copy(self.lv):div(2):exp():cmul(self.eps):cmul(gradOutput)
  return self.gradInput
end

function Sampler:_viewInput(input)
  self.mu, self.lv = input[1], input[2]
end

function Sampler:clearState()
  for k,v in ipairs(self.gradInput) do
    v:set()
  end
  return nn.utils.clear(self, 'output')
end
