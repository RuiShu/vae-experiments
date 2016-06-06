require 'image'
local DataLoader = torch.class("DataLoader")

function DataLoader:__init(batchSize, segLength, preprocess)
  self.batchSize = batchSize
  self.segLength = segLength
  self.maxAction = 9
  self.actionInput = torch.Tensor(batchSize, segLength, self.maxAction)
  self.frameInput = torch.Tensor(batchSize, segLength, 3, 210, 160)
  self:_readAllAction()
  self:_setPreprocess(preprocess)
end

function DataLoader:_readAllAction()
  self.action = torch.IntTensor(600,1000,self.maxAction):zero()
  for idx = 1,600 do
    local action = torch.load('/local-scratch/rshu15/action/action'
                                 ..idx..'.t7'):long()
    self.action[idx]:scatter(2, action:view(-1,1), 1)
  end
end

function DataLoader:_setPreprocess(preprocess)
  if preprocess == 'subMean' then
    self.mean = image.load('/local-scratch/rshu15/png/mean.png'):view(1,1,3,210,160)
    self._process = self._subMean
    self.deprocess = self._addMean
  elseif preprocess == 'None' then
    self._process = function() end
  end
end

function DataLoader:_subMean()
  self.frameInput:csub(self.mean:expandAs(self.frameInput))
end

function DataLoader:_addMean(frame)
  if frame:dim() == 3 then
    frame = frame + self.mean:view(3,210,160):expandAs(frame)
    frame[frame:gt(1)] = 1
    frame[frame:le(0)] = 0
    return frame
  end
end

function DataLoader:_fillInput()
  for i = 1,self.batchSize do
    local px = torch.random(1,500)
    local ix = torch.random(1,1000-self.segLength)
    for j = 1,self.segLength do
      ix = ix + 1
      local fp = '/local-scratch/rshu15/png/part'..px..'/image'..ix..'.png'
      local img = image.load(fp)
      self.frameInput[{i,j}]:copy(img)
      self.actionInput[{i,j}]:copy(self.action[{px,ix}])
    end
  end
end

function DataLoader:_fillValidationInput()
  for i = 1,self.batchSize do
    local px = torch.random(501,600)
    local ix = torch.random(1,1000-self.segLength)
    for j = 1,self.segLength do
      ix = ix + 1
      local fp = '/local-scratch/rshu15/png/part'..px..'/image'..ix..'.png'
      local img = image.load(fp)
      self.frameInput[{i,j}]:copy(img)
      self.actionInput[{i,j}]:copy(self.action[{px,ix}])
    end
  end
end

function DataLoader:next()
  self:_fillInput()
  self:_process()
  return {self.frameInput:view(self.batchSize, 3*self.segLength, 210, 160),
          self.actionInput}
end

function DataLoader:nextValid()
  self:_fillValidationInput()
  self:_process()
  return {self.frameInput:view(self.batchSize, 3*self.segLength, 210, 160),
          self.actionInput}
end

function DataLoader:cuda()
  require 'cunn'
  self.actionInput = self.actionInput:cuda()
  self.frameInput = self.frameInput:cuda()
  if self.mean then self.mean = self.mean:cuda() end
  return self
end

