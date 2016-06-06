require 'hdf5'
require 'utils.DataManager'
local MnistManager, parent = torch.class('MnistManager', 'DataManager')

function MnistManager:__init(batchsize)
   parent.__init(self, batchsize)
   -- get data
   local f = hdf5.open('datasets/mnist.hdf5', 'r')
   self.train = f:read('x_train'):all():double()
   self.valid = f:read('x_valid'):all():double()
   self.test = f:read('x_test'):all():double()
   self.trainLabel = f:read('t_train'):all():double()
   self.validLabel = f:read('t_valid'):all():double()
   self.testLabel = f:read('t_test'):all():double()
   f:close()
end

function MnistManager:next()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local input = self.train:index(1, v)
   local target = self.trainLabel:index(1, v)
   return {input, target}
end

function MnistManager:nextValid()
   self.current = self.current + 1
   xlua.progress(self.current, #self.indices)
   local v = self.indices[self.current]
   local input = self.valid:index(1, v)
   local target = self.validLabel:index(1, v)
   return {input, target}
end

function MnistManager:cuda()
   require 'cunn'
   self.train = self.train:cuda()
   self.valid = self.valid:cuda()
   self.test = self.test:cuda()
   self.trainLabel = self.trainLabel:cuda()
   self.validLabel = self.validLabel:cuda()
   self.testLabel = self.testLabel:cuda()
   return self
end

