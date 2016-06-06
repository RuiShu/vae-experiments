grid = {}

function grid.stack(images, nRow, nCol)
   -- nBatch x Height x Width
   if images:dim() == 3 then
      local H = images:size(2)
      local W = images:size(3)
      local grid = torch.Tensor(nRow, nCol, H, W)
      local idx = 0
      for i = 1,nRow do
         for j = 1,nCol do
            idx = idx + 1
            grid[{i,j}]:copy(images[idx])
         end
      end
      -- indexing tricks
      grid = grid:transpose(3, 4):contiguous():view(nRow, nCol*W, H)
      grid = grid:transpose(2, 3):contiguous():view(nRow*H, nCol*W)
      return grid
   elseif images:dim() == 4 then
      local C = images:size(2)
      local H = images:size(3)
      local W = images:size(4)
      local grid = torch.Tensor(nRow, nCol, C, H, W)
      local idx = 0
      for i = 1,nRow do
         for j = 1,nCol do
            idx = idx + 1
            grid[{i,j}]:copy(images[idx])
         end
      end
      -- indexing tricks
      grid = grid:transpose(2, 4):contiguous():view(nRow*H, C, nCol*W)
      grid = grid:transpose(1, 2):contiguous():view(C, nRow*H, nCol*W)
      return grid
   end
end

function grid.t(image)
   -- transpose along the last two dimensions
   local len = image:dim()
   return image:transpose(len, len-1):contiguous()
end

function grid.split(image, dim)
   -- split evenly along dim
   local N = image:size(dim)
   assert(N % 2 == 0, "Non-even number of values!")
   local mu, lv = unpack(image:split(N/2, dim))
   mu = mu:squeeze()
   lv = lv:squeeze()
   return mu, lv
end
