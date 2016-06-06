require 'nn'

local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:__init(nInputDims,sizeAverage)
  parent.__init(self)
  self.nInputDims = nInputDims or -1
  self.sizeAverage = sizeAverage or false
  -- create buffers
  self.lvDiff = torch.Tensor()
  self.qExp = torch.Tensor()
  self.muDiff = torch.Tensor()
  self.expDiff = torch.Tensor()
  self.KLD = torch.Tensor()
  self.dp = {torch.Tensor(), torch.Tensor()}
  self.dq = {torch.Tensor(), torch.Tensor()}
  self.gradInput = {self.dp, self.dq}
  self.w = 1
  self.s = 1
end

function KLDCriterion:weight(w)
  self.w = w
  return self
end

function KLDCriterion:updateOutput(p, q)
  -- p is batchSize x 2 x nDims
  -- q is batchSize x 2 x nDims
  self:_viewInput(p, q)
  self:_resizeBuffers()
  self.lvDiff:add(self.pLv, -1, self.qLv)
  self.muDiff:add(self.pMu, -1, self.qMu)
  self.expDiff:exp(self.lvDiff)
  self.qExp:mul(self.qLv, -1):exp()
  self.KLD:pow(self.muDiff, 2):cmul(self.qExp):add(self.expDiff):csub(1):csub(self.lvDiff):div(2)
  return self.KLD:sum()/self.s
end

function KLDCriterion:updateGradInput(p, q)
  self.dpMu = self.dp[1]:resizeAs(self.pMu)
  self.dpLv = self.dp[2]:resizeAs(self.pLv)
  self.dqMu = self.dq[1]:resizeAs(self.qMu)
  self.dqLv = self.dq[2]:resizeAs(self.qLv)
  -- compute dp
  self.dpMu:cmul(self.muDiff, self.qExp):mul(self.w):div(self.s)
  self.dpLv:csub(self.expDiff, 1):div(2):mul(self.w):div(self.s)
  -- compute dq
  self.dqMu:mul(self.muDiff, -1):cmul(self.qExp):mul(0):mul(self.w):div(self.s)
  self.dqLv:pow(self.muDiff, 2):cmul(self.qExp):neg():csub(self.expDiff):add(1):div(2):mul(self.w):div(self.s)
  return self.gradInput[1], self.gradInput[2]
end

function KLDCriterion:_resizeBuffers()
  -- cache from forward
  self.lvDiff:resizeAs(self.qLv)
  self.qExp:resizeAs(self.qLv)
  self.muDiff:resizeAs(self.qLv)
  self.expDiff:resizeAs(self.qLv)
  self.KLD:resizeAs(self.qLv)
end

function KLDCriterion:_viewInput(p, q)
  if self.sizeAverage then
    if p[1]:dim() == self.nInputDim then
      self.s = 1
    else
      self.s = p[1]:size(1)
    end
  end
  if not q then
    self.qZero = self.qZero or {p[1].new():resizeAs(p[1]):zero(),
                                p[2].new():resizeAs(p[2]):zero()}
    q = self.qZero
  end
  self.pMu, self.pLv = p[1], p[2]
  self.qMu, self.qLv = q[1], q[2]
end
