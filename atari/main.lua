require 'pl'
require 'optim'
require 'utils.init'
require 'models.Cnn'
require 'models.CnnCVae1'
require 'models.CnnCVae2'
require 'models.CnnCVae3'
require 'models.StackedCnn'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'

path.mkdir('save')
torch.setnumthreads(1)

local cmd = lapp[[
  --model        (default Cnn)        | the model
  --batchNorm                         | whether to use batch normalization
  --preprocess   (default subMean)    | preprocessing
  --learningRate (default 1e-4)       | learning rate
  --updateRule   (default adam)       | type of update
  --maxIter      (default 2e6)        | number of total iterations
  --iterStep     (default 1e5)        | number of iterations before each step decay
  --iterDecay    (default 0.9)        | learning rate multiplier
  --saveStep     (default 6000)       | number of steps before saving model
  --logStep      (default 100)        | number of iterations before a log
  --gpu          (default 1)          | Number of gpus used
  --unit         (default 0)          | ID of the unit if using Stacked
  --name (default '')
  ]]
cmd.ID = torch.random(1e9,4e9)
local infoList, data
if string.find(cmd.name, 'Stack') or string.find(cmd.model, 'Stack') then
  infoList = {'batchNorm', 'preprocess',
              'learningRate', 'updateRule',
              'maxIter', 'iterStep', 'iterDecay',
              'unit', 'ID'}
  data = DataManager(8,8,7,cmd.preprocess):cuda()
else
  infoList = {'batchNorm', 'preprocess',
              'learningRate', 'updateRule',
              'maxIter', 'iterStep', 'iterDecay',
              'ID'}
  data = DataManager(8,32,5,cmd.preprocess):cuda()
end
local saver = Saver(cmd, infoList)
print("Experiment: "..saver.name)

local model, state, config, iter = saver:initialize()
model:cuda()
model.network:training()
model:getParameters()
local logger = Logger(saver.name, iter, config)
data:startJobs()

while iter < cmd.maxIter do
  xlua.progress((iter % cmd.logStep)+1, cmd.logStep)
  iter = iter + 1
  local minibatch = data:next()
  local feval = function(x) return model:feval(x, minibatch) end
  optim[cmd.updateRule](feval, model.parameters, config, state)
  logger:receiveRecord(model:sendRecord(), config.learningRate)

  if iter % cmd.logStep == 0 then
    logger:log()
  end

  -- adjust learning rate and saving
  if iter % cmd.iterStep == 0 then
    config.learningRate = config.learningRate * cmd.iterDecay
    print(c.green "New learning rate:"..config.learningRate)
  end

  -- save
  if iter % cmd.saveStep == 0 then
    saver:save(model, config, state, iter)
  end
end
