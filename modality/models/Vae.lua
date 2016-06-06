local Vae = torch.class("Vae")
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function Vae:__init(struct)
  -- build model
  self.encoder, self.decoder, self.model = self:build(struct)
  self.kld = nn.KLDCriterion()
  self.rec = nn.BCECriterion()
  self.kldWeight = 1
  self.recWeight = 1
  self.rec.sizeAverage = false
  self.parameters, self.gradients = self.model:getParameters()
end

function Vae:build(struct)
  -- construct self.encoder
  local encoder = nn.Sequential()
    :add(nn.Linear(struct.x, struct.h))
    :add(nn.ReLU(true))
    :add(nn.Linear(struct.h, 2*struct.z))
    :add(nn.SplitTensor(1,1,struct.z))
  -- construct self.decoder
  local decoder = nn.Sequential()
    :add(nn.Linear(struct.z, struct.h))
    :add(nn.ReLU(true))
    :add(nn.Linear(struct.h, struct.x))
    :add(nn.Sigmoid(true))
  -- combine the two
  local input = nn.Identity()()
  local mulv = encoder(input)
  local code = nn.Sampler()(mulv)
  local recon = decoder(code)
  local model = nn.gModule({input},{mulv, recon})
  return encoder, decoder, model
end

function Vae:feval(x, minibatch)
  local input = minibatch[1]
  if self.parameters ~= x then
    self.parameters:copy(x)
  end
  self.model:zeroGradParameters()
  -- forward
  local mulv, recon = unpack(self.model:forward(input))
  local kldErr = self.kld:forward(mulv)
  local recErr = self.rec:forward(recon, input)
  -- backward
  local dmulv = self.kld:backward(mulv, pmulv)
  local drecon = self.rec:backward(recon, input)
  errorGrads = {dmulv, drecon}
  self.model:backward(input, errorGrads)
  -- record
  local nElbo = kldErr + recErr
  self.recordName = {'nElbo', 'KLD', 'REC'}
  self.record = {nElbo=nElbo/200, KLD=kldErr/200, REC=recErr/200}
  return nelbo, self.gradients
end

function Vae:sendRecord()
  local comm = {}
  comm.recordName = self.recordName
  comm.record = self.record
  return comm
end

function Vae:cuda()
  require 'cunn'
  self.model:cuda()
  self.rec:cuda()
  self.kld:cuda()
  self.parameters, self.gradients = self.model:getParameters()
  return self
end
