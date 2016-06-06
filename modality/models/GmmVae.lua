require 'models.Vae'
require 'nngraph'
require 'nnutils.init'
local GmmVae, parent = torch.class('GmmVae')

function GmmVae:__init(struct)
  -- build model
  self.prior, self.encoder, self.decoder, self.model = self:build(struct)
  self.kld = nn.GmmKLDCriterion():weight(1)
  self.rec = nn.BCECriterion()
  self.rec.sizeAverage = false
  self.parameters, self.gradients = self.model:getParameters()
end

function GmmVae:build(struct)
  -- construct self.prior
  local prior = nn.Sequential()
  prior:add(nn.Zero(1))
    :add(nn.Linear(1, 2*struct.m*struct.z))
    :add(nn.View(2*struct.m, struct.z))
    :add(nn.SplitTensor(1,2,struct.m))
  -- construct self.encoder
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(struct.x, struct.h))
    :add(nn.ReLU(true))
    :add(nn.Linear(struct.h, struct.z*2))
    :add(nn.SplitTensor(1,1,struct.z))
  -- construct self.decoder
  local decoder = nn.Sequential()
  decoder:add(nn.Linear(struct.z, struct.h))
    :add(nn.ReLU(true))
    :add(nn.Linear(struct.h, struct.x))
    :add(nn.Sigmoid(true))
  -- combine the three
  local input = nn.Identity()()
  local pmulv = prior(input)
  local mulv = encoder(input)
  local code = nn.Sampler()(mulv)
  local recon = decoder(code)
  local model = nn.gModule({input},{pmulv, mulv, recon})
  return prior, encoder, decoder, model
end

function GmmVae:feval(x, minibatch)
  local input = minibatch[1]
  if self.parameters ~= x then
    self.parameters:copy(x)
  end
  self.model:zeroGradParameters()
  -- forward
  local pmulv, mulv, recon = unpack(self.model:forward(input))
  local kldErr = self.kld:forward(mulv, pmulv)
  local recErr = self.rec:forward(recon, input)
  -- backward
  local dmulv, dpmulv = self.kld:backward(mulv, pmulv)
  local drecon = self.rec:backward(recon, input)
  error_grads = {dpmulv, dmulv, drecon}
  self.model:backward(input, error_grads)
  -- record
  local nElbo = kldErr + recErr
  self.recordName = {'nElbo', 'KLD', 'REC'}
  self.record = {nElbo=nElbo/200, KLD=kldErr/200, REC=recErr/200}
  return nElbo, self.gradients
end

function GmmVae:sendRecord()
  local comm = {}
  comm.recordName = self.recordName
  comm.record = self.record
  return comm
end

function GmmVae:cuda()
  require 'cunn'
  self.model:cuda()
  self.rec:cuda()
  self.kld:cuda()
  self.parameters, self.gradients = self.model:getParameters()
  return self
end
