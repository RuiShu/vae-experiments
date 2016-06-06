local Cnn = torch.class("Cnn")
local c = require 'trepl.colorize'
require 'nngraph'
require 'threads'

function Cnn:__init(cmd, network)
  -- build network
  self.cmd = cmd
  if cmd.gpu > 1 then
    self.network = self:parallelBuild(network)
  else
    self.network = network or self:build()
  end
  self.mse = nn.MSECriterion(false)
end

function Cnn:getNetwork()
  if self.cmd.gpu > 1 then
    return self.network:get(1)
  else
    return self.network
  end
end

function Cnn:parallelBuild(network)
  network = network or self:build()
  local gpus = {}
  for i = 1,self.cmd.gpu do gpus[i] = i end
  local paranet = nn.DataParallelTable(1, true, true)
  paranet:add(network, gpus)
  return paranet
end

function Cnn:build()
  local function SConvBReLU(network, ...)
    local arg = {...}
    network:add(nn.SpatialConvolution(...))
    if self.cmd.batchNorm then
      network:add(nn.SpatialBatchNormalization(arg[2]))
    end
    network:add(nn.ReLU(true))
  end

  local function SFConvBReLU(network, ...)
    local arg = {...}
    network:add(nn.SpatialFullConvolution(...))
    if self.cmd.batchNorm then
      network:add(nn.SpatialBatchNormalization(arg[2]))
    end
    network:add(nn.ReLU(true))
  end

  local function LinearBReLU(network, ...)
    local arg = {...}
    network:add(nn.Linear(...))
    if self.cmd.batchNorm then
      network:add(nn.BatchNormalization(arg[2]))
    end
    network:add(nn.ReLU(true))
  end

  local conv = nn.Sequential()
  SConvBReLU(conv, 12, 64, 8,8, 2,2, 1,0)
  SConvBReLU(conv, 64,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv,128,128, 6,6, 2,2, 1,1)
  SConvBReLU(conv,128,128, 4,4, 2,2, 0,0)
  conv:add(nn.View(11264))
  LinearBReLU(conv,11264,2048)
  conv:add(nn.Linear(2048,2048))

  local deconv = nn.Sequential()
  deconv:add(nn.Linear(2048,2048))
  LinearBReLU(deconv,2048,11264)
  deconv:add(nn.View(128,11,8))
  SFConvBReLU(deconv,128,128, 4,4, 2,2, 0,0)
  SFConvBReLU(deconv,128,128, 6,6, 2,2, 1,1)
  SFConvBReLU(deconv,128,128, 6,6, 2,2, 1,1)
  deconv:add(nn.SpatialFullConvolution(128,3, 8,8, 2,2, 1,0))

  local action = nn.Sequential()
    :add(nn.Linear(9,2048))

  local network = nn.Sequential()
    :add(nn.ParallelTable()
           :add(conv)
           :add(action))
    :add(nn.CMulTable())
    :add(deconv)

  return network
end

function Cnn:feval(x, minibatch)
  local frameBatch = minibatch[1]
  local action = minibatch[2][{{},4}]
  local input = frameBatch[{{},{1,12}}]
  local target = frameBatch[{{},{13,15}}]
  local batchSize = input:size(1)
  if self.parameters ~= x then
    self.parameters:copy(x)
  end
  self.network:zeroGradParameters()
  -- forward
  local pred = self.network:forward({input, action})
  local predErr = self.mse:forward(pred, target)/batchSize
  -- backward
  local dpred = self.mse:backward(pred, target)
  self.network:backward({input, action}, dpred:div(batchSize))
  -- record
  self.recordName = {'predErr'}
  self.record = {predErr=predErr}
  return predErr, self.gradients
end

function Cnn:sendRecord()
  local comm = {}
  comm.recordName = self.recordName
  comm.record = self.record
  return comm
end

function Cnn:cuda()
  require 'cunn'
  self.network:cuda()
  self.mse:cuda()
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.fastest = true
  cudnn.convert(self.network, cudnn)
  return self
end

function Cnn:getParameters()
  self.parameters, self.gradients = self.network:getParameters()
  return self.parameters, self.gradients
end
