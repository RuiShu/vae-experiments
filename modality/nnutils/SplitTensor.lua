local SplitTensor, parent = torch.class('nn.SplitTensor', 'nn.Module')

function SplitTensor:__init(dimension, nInputDims, splitSize)
   parent.__init(self)
   self.dimension = dimension
   self.nInputDims = nInputDims
   self.splitSize = splitSize or 1
end

function SplitTensor:_getPositiveDimension(input)
   local dimension = self.dimension
   if dimension < 0 then
      dimension = input:dim() + dimension + 1
   elseif self.nInputDims and input:dim()==(self.nInputDims+1) then
      dimension = dimension + 1
   end
   return dimension
end

function SplitTensor:updateOutput(input)
   local dimension = self:_getPositiveDimension(input)
   self.output = input:split(self.splitSize,dimension)
   return self.output
end

function SplitTensor:updateGradInput(input, gradOutput)
   local dimension = self:_getPositiveDimension(input)
   if self.gradInput then
      self.gradInput:resizeAs(input)
      self.gradInput:cat(gradOutput,dimension)
   end
   return self.gradInput
end
