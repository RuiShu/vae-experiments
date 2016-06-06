require 'hdf5'

local DataManager = torch.class("DataManager")

function DataManager:__init(batchSize)
   self.batchSize = batchSize
   self.indices = nil
   self.current = nil
   self.train = nil
   self.valid = nil
   self.test = nil
end

function DataManager:inEpoch()
   return self.current ~= #self.indices
end

function DataManager:shuffle(batchSize)
   self.batchSize = batchSize or self.batchSize
   self.indices = torch.randperm(self.train:size(1)):long():split(self.batchSize)
   self.indices[#self.indices] = nil
   self.current = 0
end

function DataManager:shuffleValid(validSize)
   self.validSize = validSize or self.validSize or self.batchSize
   self.indices = torch.randperm(self.valid:size(1)):long():split(self.validSize)
   self.indices[#self.indices] = nil
   self.current = 0
end

function DataManager:next()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local input = self.train:index(1, v)
   return {input}
end

function DataManager:nextValid()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local input = self.valid:index(1, v)
   return {input}
end

function DataManager:cuda()
   require 'cunn'
   self.train = self.train:cuda()
   return self
end

