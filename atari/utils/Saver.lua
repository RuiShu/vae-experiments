require 'lfs'
local Saver = torch.class("Saver")
local c = require 'trepl.colorize'

function Saver:__init(cmd, infoList)
  self.cmd = cmd
  self.infoList = infoList
  if cmd.name ~= "" then self:__updateCmd(cmd.name) end
  self.name = self:__createName()
end

function Saver:__createName()
  local name = self.cmd.model
  for k,v in pairs(self.infoList) do
    name = name..'_'..v..'='..tostring(self.cmd[v])
  end
  return name
end

function Saver:__updateCmd(name)
  name = 'model='..name
  local info = split(name, '_')
  for _,v in ipairs(info) do
    for k,_ in pairs(self.cmd) do
      local s,e = string.find(v,k)
      if e then
        if type(self.cmd[k]) == 'boolean' then
          self.cmd[k] = string.sub(v,e+2) == 'true'
        elseif type(self.cmd[k]) == 'number' then
          self.cmd[k] = tonumber(string.sub(v,e+2))
        else
          self.cmd[k] = string.sub(v,e+2)
        end
      end
    end
  end
end

function Saver:initialize()
  local network, model, state, config, iter
  if self.cmd.name ~= '' then
    local save = torch.load('save/'..self.cmd.name..'.t7')
    network = save.network
    state = save.state
    config = save.config
    iter = save.iter
    model = _G[self.cmd.model](self.cmd, network)
  else
    network = self:loadNetwork()
    state = {}
    config = {learningRate=self.cmd.learningRate}
    iter = 0
    model = _G[self.cmd.model](self.cmd, network)
  end
  return model, state, config, iter
end

function Saver:loadNetwork()
  if string.find(self.cmd.model, 'Stack') then
    assert(self.cmd.unit ~= 0, 'Must specify unit ID for Stacked architecture')
    for f in lfs.dir('save') do
      if string.match(f, 'ID='..tostring(self.cmd.unit)) then
        local save = torch.load('save/'..f)
        return save.network
      end
    end
  end
end

function Saver:save(model, config, state, iter)
  local network = model:getNetwork()
  network:clearState()
  collectgarbage()
  collectgarbage()
  local file = 'save/'..self.name..'.t7'
  torch.save(file, {config=config,
                    state=state,
                    network=network,
                    iter=iter})
  print(c.green "Saved checkpoint to "..file)
  sys.execute('cp '..file..' '..file..'.bak')
  print(c.green "Saved backup")
end
