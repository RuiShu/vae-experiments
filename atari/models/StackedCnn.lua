local StackedCnn = torch.class("StackedCnn")
local c = require 'trepl.colorize'
require 'nnutils.init'
require 'nngraph'

function StackedCnn:__init(cmd, unit)
  -- build network
  self.unit = unit
  self.cmd = cmd
  if cmd.gpu > 1 then
    self.network = self:parallelBuild(unit)
  else
    self.network = self:build(unit)
  end
  self.mse = nn.MSECriterion(false)
end

function StackedCnn:getNetwork()
  return self.unit
end

function StackedCnn:parallelBuild(unit)
  local network = self:build(unit)
  local gpus = {}
  for i = 1,self.cmd.gpu do gpus[i] = i end
  local paranet = nn.DataParallelTable(1, true, true)
  paranet:add(network, gpus)
  return paranet
end

function StackedCnn:build(unit)
  -- create shift
  local index = torch.linspace(4,12,9):long()
  local shift = nn.Sequential()
    :add(nn.ParallelTable()
           :add(nn.FixedIndex(1, index, 3))
           :add(nn.Identity()))
    :add(nn.JoinTable(1,3))
  -- make clones
  local unit1 = unit
  local unit2 = unit:clone('weight','bias','gradWeight','gradBias', 'running_mean', 'running_var')
  local unit3 = unit:clone('weight','bias','gradWeight','gradBias', 'running_mean', 'running_var')
  local shift1 = shift:clone('weight','bias','gradWeight','gradBias')
  local shift2 = shift:clone('weight','bias','gradWeight','gradBias')
  -- make gmodule
  local i1 = nn.Identity()()
  local a1 = nn.Identity()()
  local a2 = nn.Identity()()
  local a3 = nn.Identity()()
  local o1 = unit1({i1,a1})
  local i2 = shift1({i1,o1})
  local o2 = unit2({i2,a2})
  local i3 = shift2({i2,o2})
  local o3 = unit3({i3,a3})

  local network = nn.gModule({i1,a1,a2,a3},{o1,o2,o3})
  return network
end

function StackedCnn:feval(x, minibatch)
  local frameBatch = minibatch[1]
  local input = frameBatch[{{},{1,12}}]
  local a1 = minibatch[2][{{},4}]
  local a2 = minibatch[2][{{},5}]
  local a3 = minibatch[2][{{},6}]
  local target = frameBatch[{{},{13,21}}]
  local batchSize = input:size(1)
  local kSize = 3
  if self.parameters ~= x then
    self.parameters:copy(x)
  end
  self.network:zeroGradParameters()
  -- forward
  local p1, p2, p3 = unpack(self.network:forward({input, a1, a2, a3}))
  local pe1 = self.mse:forward(p1, target[{{},{1,3}}])/batchSize/kSize
  local pe2 = self.mse:forward(p2, target[{{},{4,6}}])/batchSize/kSize
  local pe3 = self.mse:forward(p3, target[{{},{7,9}}])/batchSize/kSize
  -- backward
  local dpe1 = self.mse:backward(p1, target[{{},{1,3}}]):div(batchSize):div(kSize)
  local dpe2 = self.mse:backward(p2, target[{{},{4,6}}]):div(batchSize):div(kSize)
  local dpe3 = self.mse:backward(p3, target[{{},{7,9}}]):div(batchSize):div(kSize)
  self.network:backward({input, a1, a2, a3}, {dpe1, dpe2, dpe3})
  -- record
  self.recordName = {'predErr', 'predErr1', 'predErr2', 'predErr3'}
  self.record = {predErr=pe1+pe2+pe3, predErr1=pe1*3, predErr2=pe2*3, predErr3=pe3*3}
  return predErr, self.gradients
end

function StackedCnn:sendRecord()
  local comm = {}
  comm.recordName = self.recordName
  comm.record = self.record
  return comm
end

function StackedCnn:cuda()
  require 'cunn'
  self.network:cuda()
  self.mse:cuda()
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.fastest = true
  cudnn.convert(self.network, cudnn)
  return self
end

function StackedCnn:getParameters()
  self.parameters, self.gradients = self.network:getParameters()
  return self.parameters, self.gradients
end
