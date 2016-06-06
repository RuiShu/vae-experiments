require 'optim'
local Logger = torch.class("Logger")
local c = require 'trepl.colorize'

function Logger:__init(name, iter)
  self.iter = iter or 0
  self.name = name
  -- create logger
  while io.open('save/log/'..name..'.log','r') do
    name = name..'C'
  end
  self.optLogger = optim.Logger('save/log/'..name..'.log')
end

function Logger:receiveRecord(comm, learningRate)
  self.learningRate = learningRate
  self.iter = self.iter + 1
  if not self.optNames then
    self.optNames = {'iter'}
    for _,v in ipairs(comm.recordName) do
      table.insert(self.optNames, v)
    end
    table.insert(self.optNames, 'learningRate')
    self.optLogger:setNames(self.optNames)
  end

  for _,k in ipairs(comm.recordName) do
    self[k] = self[k] or comm.record[k]
    self[k] = self[k]*0.99 + comm.record[k]*0.01
  end

  self.record = {}
  for _,k in pairs(self.optNames) do
    table.insert(self.record, self[k])
  end
  self.optLogger:add(self.record)
end

function Logger:log()
  print(c.green ' Experiment: '..self.name)
  for _,k in pairs(self.optNames) do
    print(c.red '==> '..k..': '..self[k])
  end
end

