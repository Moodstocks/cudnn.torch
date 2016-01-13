local SpatialAveragePooling, parent
   = torch.class('cudnn.SpatialAveragePooling', 'cudnn._Pooling')

local function backwardCompatible(self)
   if self.ceil_mode == nil then
      self.ceil_mode = false
      self.count_include_pad = true
      self.padH = 0
      self.padW = 0
   end
end

function SpatialAveragePooling:setCountIncludePad()
   self.count_include_pad = true
   return self
end

function SpatialAveragePooling:setCountExcludePad()
   self.count_include_pad = false
   return self
end

function SpatialAveragePooling:updateOutput(input)
   -- for nn <> cudnn conversion
   backwardCompatible(self)
   if self.divide ~= nil then
      assert(self.divide, 'not supported')
   end

   if self.count_include_pad == nil then
      self.count_include_pad = true
   end
   if self.count_include_pad then
      self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
   else 
      self.mode = 'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING'
   end
   return parent.updateOutput(self, input)
end

function SpatialAveragePooling:__tostring__()
   return nn.SpatialAveragePooling.__tostring__(self)
end
