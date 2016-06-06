require 'pl'
require 'optim'
require 'utils.MnistManager'
require 'utils.Logger'
require 'models.init'
require 'cunn'
path.mkdir('save')
local c = require 'trepl.colorize'

local cmd = lapp[[
  --gpu          (default 1)     | 1 if gpu, 0 if not gpu
  --model        (default Vae)   | which model to use
  --hSize        (default 400)   | size of hidden layer
  --zSize        (default 10)    | size of latent layer
  --mSize        (default 1000)  | number of mixtures
  --learningRate (default 0.001) | learning rate
  --maxEpoch     (default 400)   | number of total epochs
  --epochStep    (default 100)   | number of steps before each step decay
  --epochDecay   (default 0.1)   | epoch step decay rate
  --saveStep     (default 100)    | number of steps before each save
  --showVis                      | show training visualization
  ]]

local saveInfo = {'hSize', 'zSize', 'mSize', 'learningRate',
                  'maxEpoch', 'epochStep', 'epochDecay'}
local name = cmd.model
for k,v in pairs(saveInfo) do  name = name..'_'..v..'='..tostring(cmd[v]) end
name = name..'_ID='..torch.random(1e9,4e9)
print("Experiment: "..name)

local model = _G[cmd.model]({x = 784, h = cmd.hSize, z = cmd.zSize, m = cmd.mSize})
local state = {}
local config = {learningRate = cmd.learningRate}
local iter = 0
local epoch = 0

local data = MnistManager(200)
local logger = Logger(name, iter)

data:cuda()
model:cuda()

while epoch < cmd.maxEpoch do
  -- training
  epoch = epoch + 1
  data:shuffle()
  while data:inEpoch() do
    local minibatch = data:next()
    local feval = function(x) return model:feval(x, minibatch) end
    optim.adam(feval, model.parameters, config, state)
    logger:receiveRecord(model:sendRecord(), config.learningRate)
  end
  logger:log()
  -- adjust learning rate and saving
  if epoch % cmd.epochStep == 0 then
    config.learningRate = config.learningRate * cmd.epochDecay
    print(c.green "New learning rate:"..config.learningRate)
  end
  if epoch % cmd.saveStep == 0 then
    local file = 'save/'..name..'.t7'
    torch.save(file, {config=config, state=state, model=model})
    print(c.green "Saved checkpoint to "..file)
  end
end
