require 'nn'

local GaussianCriterion, parent = torch.class('nn.GaussianCriterion', 'nn.Criterion')

function GaussianCriterion:__init(nInputDims, sizeAverage)
  parent.__init(self)
  self.nInputDims = nInputDims or -1
  self.sizeAverage = sizeAverage or false
  -- create buffer
  self.diff = torch.Tensor()
  self.pExp = torch.Tensor()
  self.diffExp = torch.Tensor()
  self.expElem = torch.Tensor()
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.w = 1
  self.s = 1
end

function GaussianCriterion:weight(weight)
  self.w = weight
  return self
end

function GaussianCriterion:updateOutput(p, x)
  -- negative log likelihood using p as gaussian distribution hypothesis
  -- and x as observation
  self:_viewInput(p, x)
  self:_resizeBuffers()
  self.diff:csub(x, self.pMu)
  self.pExp:mul(self.pLv, -1):exp()
  self.diffExp:pow(self.diff, 2):cmul(self.pExp)
  self.expElem:add(self.diffExp, self.pLv):add(math.log(2*math.pi))
  self.output = self.expElem:sum()*0.5
  return self.output/self.s
end

function GaussianCriterion:updateGradInput(p, x)
  self.dpMu = self.gradInput[1]:resizeAs(self.pMu)
  self.dpLv = self.gradInput[2]:resizeAs(self.pLv)
  self.dpMu:cmul(self.diff, self.pExp):neg():mul(self.w):div(self.s)
  self.dpLv:csub(self.diffExp, 1):div(2):neg():mul(self.w):div(self.s)
  return self.gradInput
end

function GaussianCriterion:_resizeBuffers()
  self.diff:resizeAs(self.pMu)
  self.pExp:resizeAs(self.pMu)
  self.diffExp:resizeAs(self.pMu)
  self.expElem:resizeAs(self.pMu)
end

function GaussianCriterion:_viewInput(p, x)
  if self.sizeAverage then
    if p[1]:dim() == self.nInputDim then
      self.s = 1
    else
      self.s = p[1]:size(1)
    end
  end
  self.pMu, self.pLv = p[1], p[2]
end
