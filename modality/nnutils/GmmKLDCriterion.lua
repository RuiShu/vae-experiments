require 'nn'

local GmmKLDCriterion, parent = torch.class('nn.GmmKLDCriterion', 'nn.Criterion')

function GmmKLDCriterion:__init()
  parent.__init(self)
  -- create buffers
  self.lvDiff = torch.Tensor()
  self.qExp = torch.Tensor()
  self.muDiff = torch.Tensor()
  self.expDiff = torch.Tensor()
  self.expElem = torch.Tensor()
  self.expSum = torch.Tensor()
  self.preOut = torch.Tensor()
  self.dpMuBuf = torch.Tensor()
  self.dpLvBuf = torch.Tensor()
  self.preKld = torch.Tensor()
  self.shift = torch.Tensor()
  self.dp = {torch.Tensor(), torch.Tensor()}
  self.dq = {torch.Tensor(), torch.Tensor()}
  self.gradInput = {self.dp, self.dq}
  self.w = 1
  self.pMu = torch.Tensor()
  self.pLv = torch.Tensor()
end

function GmmKLDCriterion:weight(w)
  self.w = w
  return self
end

function GmmKLDCriterion:updateOutput(p, q)
  -- p is batchSize x 2 x nDims
  -- q is batchSize x (2*nMixtures) x nDims
  self:_viewInput(p,q)
  self:_resizeBuffers()
  self.pMu = self.pMu:expandAs(self.qMu)
  self.pLv = self.pLv:expandAs(self.qLv)
  self.lvDiff:add(self.pLv, -1, self.qLv)
  self.muDiff:add(self.pMu, -1, self.qMu)
  self.expDiff:exp(self.lvDiff)
  self.qExp:mul(self.qLv, -1):exp()
  self.preKld:pow(self.muDiff, 2):cmul(self.qExp):add(self.expDiff):csub(1):csub(self.lvDiff)
  -- prevent overflow/underflow
  self.expElem:sum(self.preKld, self.len):div(2)
  self.shift:min(self.expElem, self.len-1)
  self.shiftEx = self.shift:expandAs(self.expElem)
  self.expElem:neg():add(self.shiftEx):exp()
  self.expSum:sum(self.expElem, self.len-1)
  self.preOut:div(self.expSum, self.nMix):log():neg():add(self.shift)
  self.output = self.preOut:sum()
  return self.output
end

function GmmKLDCriterion:updateGradInput(p, q)
  self.dpMu = self.dp[1]:resizeAs(self.pMu)
  self.dpLv = self.dp[2]:resizeAs(self.pLv)
  self.dqMu = self.dq[1]:resizeAs(self.qMu)
  self.dqLv = self.dq[2]:resizeAs(self.qLv)
  self.expSum = self.expSum:expandAs(self.expElem)
  self.expElem:cdiv(self.expSum)
  self.expElem = self.expElem:expandAs(self.qExp)
  -- compute dp
  self.dpMuBuf:cmul(self.muDiff, self.qExp):cmul(self.expElem)
  self.dpLvBuf:csub(self.expDiff, 1):div(2):cmul(self.expElem)
  self.dpMu:sum(self.dpMuBuf, self.len-1):mul(self.w)
  self.dpLv:sum(self.dpLvBuf, self.len-1):mul(self.w)
  -- compute dq
  self.dqMu:mul(self.muDiff, -1):cmul(self.qExp):cmul(self.expElem):mul(self.w)
  self.dqLv:pow(self.muDiff, 2):cmul(self.qExp):neg():csub(self.expDiff):add(1):div(2):cmul(self.expElem):mul(self.w)
  self.dpMu:resizeAs(p[1])
  self.dpLv:resizeAs(p[2])
  return self.gradInput[1], self.gradInput[2]
end

function GmmKLDCriterion:_resizeBuffers()
  self.lvDiff:resizeAs(self.qLv)
  self.qExp:resizeAs(self.qLv)
  self.muDiff:resizeAs(self.qLv)
  self.expDiff:resizeAs(self.qLv)
  self.preKld:resizeAs(self.qLv)
  local size = self.qLv:size()
  size[#size] = 1
  self.expElem:resize(size)
  size[#size-1] = 1
  self.expSum:resize(size)
  self.preOut:resize(size)
  self.shift:resize(size)
  self.dpMuBuf:resizeAs(self.qLv)
  self.dpLvBuf:resizeAs(self.qLv)
end

function GmmKLDCriterion:_viewInput(p,q)
  -- nBatch x nMix x nDims or nMix x nDims
  self.len = q[1]:dim()
  self.nMix = q[1]:size(self.len-1)
  local size = q[1]:size(); size[#size-1] = 1
  self.qMu, self.qLv = q[1], q[2]
  self.pMu:resize(size):copy(p[1])
  self.pLv:resize(size):copy(p[2])
end
