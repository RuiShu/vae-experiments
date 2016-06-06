require 'nn'

local Sampler, parent = torch.class('nn.GmmSampler', 'nn.Module')

function Sampler:__init()
   parent.__init(self)
   self.eps = torch.Tensor()
   self.muBuf = torch.Tensor()
   self.lvBuf = torch.Tensor()
   self.prob = torch.Tensor()
   self.idx = torch.LongTensor()
   self.lv = torch.Tensor()
   self.mu = torch.Tensor()
end 

function Sampler:updateOutput(input)
   -- input: batchsize x 2*nMix x nDims
   self:_viewInput(input)
   self.eps:resizeAs(self.lv):copy(torch.randn(self.lv:size()))
   self.output:resizeAs(self.lv):copy(self.lv)
   self.output:div(2):exp():cmul(self.eps):add(self.mu)
   return self.output
end

function Sampler:_viewInput(input)
   self.len = input:dim()
   self.nMix = input:size(self.len-1)/2
   self.muOrig, self.lvOrig = unpack(input:split(self.nMix, self.len-1))
   if self.len == 3 then
      local nBatch = input:size(1)
      local nDim = input:size(3)
      self.prob:resize(nBatch, self.nMix):fill(1/self.nMix)
      self.idx:resize(nBatch, 1)
      self.prob.multinomial(self.idx, self.prob, 1)
      self.idx = self.idx:view(nBatch, 1, 1):expand(nBatch, 1, nDim)
      self.mu:resize(nBatch, 1, nDim)
      self.lv:resize(nBatch, 1, nDim)
      self.mu:gather(self.muOrig, 2, self.idx)
      self.lv:gather(self.lvOrig, 2, self.idx)
      self.mu = self.mu:view(nBatch, nDim)
      self.lv = self.lv:view(nBatch, nDim)
   end
end
