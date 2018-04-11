require 'torch'
require 'nn'
require 'image'
require 'hdf5'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint', './data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.4)
cmd:option('-num_proposals', 1000)
cmd:option('-boxes_per_image', 100)

cmd:option('-input_txt', '')
cmd:option('-max_images', 0)
cmd:option('-output_h5', '')

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)


local function run_image(model, img_path, opt, dtype)
  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  local boxes_xcycwh, feats = model:extractFeatures(img_caffe:type(dtype))
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  return boxes_xywh, feats
end
print("done run image")

local function main()
  local opt = cmd:parse(arg)
  assert(opt.input_txt ~= '', 'Must provide -input_txt')
  assert(opt.output_h5 ~= '', 'Must provide -output_h5')
  
  -- Read the text file of image paths
  local image_paths = {}
  for image_path in io.lines(opt.input_txt) do
    table.insert(image_paths, image_path)
    if opt.max_images > 0 and #image_paths == opt.max_images then
      break
    end
  end
  print("done check image")
    
  -- Load and set up the model
  local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
  local checkpoint = torch.load(opt.checkpoint)
  local model = checkpoint.model
  model:convert(dtype, use_cudnn)
  model:setTestArgs{
    rpn_nms_thresh = opt.rpn_nms_thresh,
    final_nms_thresh = opt.final_nms_thresh,
    num_proposals = opt.num_proposals,
  }
  print("done set test args")
  model:evaluate()
  
  -- Set up the output tensors
  -- torch-hdf5 can only create datasets from tensors; there is no support for
  -- creating a dataset and then writing it in pieces. Therefore we have to
  -- keep everything in memory and then write it to disk ... gross.
  -- 13k images, 100 boxes per image, and 4096-dimensional features per box
  -- will take about 20GB of memory.
  local N = #image_paths
  local M = opt.boxes_per_image
  local D = 4096 -- TODO this is specific to VG
  local all_boxes = torch.FloatTensor(N, M, 4):zero()
  local all_feats = torch.FloatTensor(N, M, D):zero()
  
  -- Actually run the model
  for i, image_path in ipairs(image_paths) do
    print(string.format('Processing image %d / %d', i, N))
    local boxes, feats = run_image(model, image_path, opt, dtype)
    all_boxes[i]:copy(boxes[{{1, M}}])
    all_feats[i]:copy(feats[{{1, M}}])
  end
  print("done processing image")
  
  print("start making data file")
  -- Write data to disk, instead of HDF5 files
  
 
  file = torch.DiskFile('opt.output_h5' .. '-feats', 'w')
  file:writeObject(all_feats)
  file:close()
  print("done writing features")


  file = torch.DiskFile('opt.output_h5' .. '-boxes', 'w')
  file:writeObject(all_boxes)
  file:close()
  print("done writing boxes")


end

main()
