require 'image'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')
local DataManager = torch.class("DataManager")

function DataManager:__init(nThread, batchSize, segLength, preprocess)
  self:_setPreprocess(preprocess)
  self.nThread = nThread
  self.pool = threads.Threads(
    nThread,
    function()
      torch.setnumthreads(1)
      require 'utils.DataLoader'
    end,
    function()
      data = DataLoader(batchSize, segLength, preprocess)
    end
  )
end

function DataManager:_setPreprocess(preprocess)
  if preprocess == 'subMean' then
    self.mean = image.load('/local-scratch/rshu15/png/mean.png'):view(1,1,3,210,160)
    self.process = self._subMean
    self.deprocess = self._addMean
  elseif preprocess == 'None' then
    self.process = function() end
  end
end

function DataManager:_subMean(frame)
  if frame:dim() == 3 then
    return frame - self.mean:view(3,210,160):expandAs(frame)
    -- return torch.csub(frame, self.mean:view(3,210,160):expandAs(frame))
  end
end

function DataManager:_addMean(frame)
  if frame:dim() == 3 then
    frame = frame + self.mean:view(3,210,160):expandAs(frame)
    frame[frame:gt(1)] = 1
    frame[frame:le(0)] = 0
    return frame
  end
end

function DataManager:startJobs()
  self.requestData = function()
    local minibatch = data:next()
    return minibatch
  end
  self.pushData = function(minibatch)
    self.result = {minibatch[1]:type(self.type),
                   minibatch[2]:type(self.type)}
  end
  for i = 1,self.nThread do
    self.pool:addjob(self.requestData, self.pushData)
  end
end

function DataManager:next()
  self.pool:dojob()
  self.pool:addjob(self.requestData, self.pushData)
  return self.result
end

function DataManager:cuda()
  require 'cunn'
  self.type = 'torch.CudaTensor'
  self.mean = self.mean:cuda()
  return self
end
