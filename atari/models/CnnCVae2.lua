local c = require 'trepl.colorize'
require 'nngraph'
require 'threads'
require 'nnutils'
local CnnCVae2 = torch.class("CnnCVae2")

function CnnCVae2:__init(cmd, network)
  -- build network
  self.cmd = cmd
  if cmd.gpu > 1 then
    self.network = self:parallelBuild(network)
  else
    self.network = network or self:build()
  end
  self.kld = nn.KLDCriterion(1,true)
  self.rec = nn.MSECriterion(false)
end

function CnnCVae2:getNetwork()
  if self.cmd.gpu > 1 then
    return self.network:get(1)
  else
    return self.network
  end
end

function CnnCVae2:parallelBuild(network)
  network = network or self:build()
  gpus = {}
  for i = 1,self.cmd.gpu do gpus[i] = i end
  local paranet = nn.DataParallelTable(1, true, true)
  paranet:add(network, gpus)
  return paranet
end

function CnnCVae2:build()
  local function SConvBReLU(network, ...)
    local arg = {...}
    network:add(nn.SpatialConvolution(...))
    network:add(nn.SpatialBatchNormalization(arg[2]))
    network:add(nn.ReLU(true))
  end

  local function SFConvBReLU(network, ...)
    local arg = {...}
    network:add(nn.SpatialFullConvolution(...))
    network:add(nn.SpatialBatchNormalization(arg[2]))
    network:add(nn.ReLU(true))
  end

  local function LinearBReLU(network, ...)
    local arg = {...}
    network:add(nn.Linear(...))
    network:add(nn.BatchNormalization(arg[2]))
    network:add(nn.ReLU(true))
  end

  local action = nn.Linear(9,1024)

  local conv = nn.Sequential()
  SConvBReLU(conv, 12, 64, 8,8, 2,2, 1,0)
  SConvBReLU(conv, 64,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv,128,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv,128,128, 4,4, 2,2, 0,1)
  SConvBReLU(conv,128,128, 2,2, 2,2, 0,0)
  conv:add(nn.View(3072))
  LinearBReLU(conv,3072,2048)
  conv:add(nn.Linear(2048,1024))

  local conv2 = nn.Sequential()
  SConvBReLU(conv2, 3, 64, 8,8, 2,2, 1,0)
  SConvBReLU(conv2, 64,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv2,128,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv2,128,128, 4,4, 2,2, 0,1)
  SConvBReLU(conv2,128,128, 2,2, 2,2, 0,0)
  conv2:add(nn.View(3072))
  LinearBReLU(conv2,3072,2048)
  conv2:add(nn.Linear(2048,1024))

  local prior = nn.Sequential()
  LinearBReLU(prior,1024,512)
  LinearBReLU(prior,512,256)
  prior:add(nn.Linear(256,128))
  prior:add(nn.SplitTensor(1,1,64))

  local encoder = nn.Sequential()
  LinearBReLU(encoder,2048,512)
  LinearBReLU(encoder,512,256)
  encoder:add(nn.Linear(256,128))
  encoder:add(nn.SplitTensor(1,1,64))

  local decoder = nn.Sequential()
  LinearBReLU(decoder,64,256)
  LinearBReLU(decoder,256,512)
  decoder:add(nn.Linear(512,1024))

  local deconv = nn.Sequential()
  LinearBReLU(deconv,2048,2048)
  LinearBReLU(deconv,2048,3072)
  deconv:add(nn.View(128,6,4))
  SFConvBReLU(deconv,128,128, 2,2, 2,2, 0,0)
  SFConvBReLU(deconv,128,128, 4,4, 2,2, 0,1)
  SFConvBReLU(deconv,128,128, 6,6, 2,2, 1,1)
  SFConvBReLU(deconv,128, 64, 6,6, 2,2, 1,1)
  deconv:add(nn.SpatialFullConvolution(64,3, 8,8, 2,2, 1,0))

  local hadamard = nn.CMulTable()
  local concat1 = nn.JoinTable(1,1)
  local concat2 = nn.JoinTable(1,1)
  local sampler = nn.Sampler()

  -- x is current frame, a is current action, y is future frame
  local x = nn.Identity()()
  local a = nn.Identity()()
  local y = nn.Identity()()
  -- compute hidden states
  local hx = conv(x)
  local hy = conv2(y)
  local ha = action(a)
  local hxa = hadamard({hx, ha})
  local hxay = concat1({hxa, hy})
  -- CVAE
  local muLv = encoder(hxay)
  local pmuLv = prior(hx)
  local c = sampler(muLv)
  local hc = decoder(c)
  local hxac = concat2({hxa, hc})
  local pred = deconv(hxac)
  -- combine
  local network = nn.gModule({x, a, y}, {pmuLv, muLv, pred})
  return network
end

function CnnCVae2:feval(x, minibatch)
  local frameBatch = minibatch[1]:contiguous()
  local action = minibatch[2][{{},4}]:contiguous()
  local input = frameBatch[{{},{1,12}}]:contiguous()
  local target = frameBatch[{{},{13,15}}]:contiguous()
  local batchSize = input:size(1)
  if self.parameters ~= x then
    self.parameters:copy(x)
  end
  self.network:zeroGradParameters()
  local pmulv, mulv, pred = unpack(self.network:forward({input, action, target}))
  local kldErr = self.kld:forward(mulv, pmulv)
  local recErr = self.rec:forward(pred, target)/batchSize
  local nElbo = kldErr + recErr
  local dmulv, dpmulv = self.kld:backward(mulv, pmulv)
  local dpred = self.rec:backward(pred, target)
  local gradOut = {dpmulv, dmulv, dpred:div(batchSize)}
  self.network:backward({input, action, target}, gradOut)
  -- record
  self.recordName = {'nElbo', 'KLD', 'REC'}
  self.record = {nElbo=nElbo, KLD=kldErr, REC=recErr}
  return nElbo, self.gradients
end

function CnnCVae2:sendRecord()
  local comm = {}
  comm.recordName = self.recordName
  comm.record = self.record
  return comm
end

function CnnCVae2:cuda()
  require 'cunn'
  self.network:cuda()
  self.kld:cuda()
  self.rec:cuda()
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.fastest = true
  cudnn.convert(self.network, cudnn)
  return self
end

function CnnCVae2:getParameters()
  self.parameters, self.gradients = self.network:getParameters()
  return self.parameters, self.gradients
end
