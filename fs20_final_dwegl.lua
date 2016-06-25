-- final_14.lua
------------------------------------------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'unsup'   -- for using 'kmeans'
------------------------------------------------------------------------------------------------------
-- global parameters
local ParamBank = require 'ParamBank'
local label     = require 'fs20_subset_classes'
-- print(label)

local SpatialConvolution = nn.SpatialConvolution
local SpatialConvolutionMM = nn.SpatialConvolutionMM
local SpatialMaxPooling = nn.SpatialMaxPooling

local cuda = true;

if cuda then
   require 'cunn'
   require 'cudnn'
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialConvolutionMM = cudnn.SpatialConvolution
   SpatialMaxPooling = cudnn.SpatialMaxPooling
end

dataset_code = 'fs20'
experiment_code = 'dwegl'
total_classes = 20
-- training/test size
trsize = 2000
tesize = 1130
-- type of OverFeat conv-net
local network  = 'big'
-- local batchSize = 8
-- flag denoting when to begin active-learning
active_learning_flag = false
epoch_per_update = 4
epoch_counter = 0
initial_training_counter = 20
-- samples_per_class = trsize / total_classes -- depends on the configuration of the training metadata
samples_added_per_update = total_classes
optimization_factor = 1.0
training_threshold = trsize
initial_samples_per_class = 4
labeled_pool_samples = {}
labeled_pool_labels = {}
current_pool_size = 0
egl_top_k_count = 5
k_means_iterations = 100

-- system parameters
local threads = torch.getnumthreads()
local offset  = 0

torch.setdefaulttensortype('torch.FloatTensor')
tensorType = 'float'
if cuda then tensorType = 'cuda' end

torch.setnumthreads(threads)
print('==> #threads:', threads)
---------------------------------------------------------------------------------------------------------
print '==> processing options'

--cmd = torch.CmdLine()
--cmd:text()
--cmd:text('SVHN Loss Function')
--cmd:text()
--cmd:text('Options:')
---- global:
--cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
--cmd:option('-threads', 2, 'number of threads')
---- data:
--cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
---- model:
--cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
---- loss:
--cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
---- training:
--cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
--cmd:option('-plot', false, 'live plot')
--cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
--cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
--cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
--cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
--cmd:option('-momentum', 0, 'momentum (SGD only)')
--cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
--cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
--cmd:option('-type', 'double', 'type: double | float | cuda')
--cmd:text()
--opt = cmd:parse(arg or {})

opt = lapp[[
        -a,--seed               (default 1)             fixed input seed for repeatable experiments
        -b,--size               (default full)          how many samples do we load: small | full | extra
        -c,--model              (default convnet)       type of model to construct: linear | mlp | convnet
        -d,--loss               (default nll)           type of loss function to minimize: nll | mse | margin
        -e,--plot               (default false)         live plot
        -f,--optimization       (default SGD)           optimization method: SGD | ASGD | CG | LBFGS
        -s,--save               (default "results")     subdirectory to save logs
        -b,--batchSize          (default 8)             batch size
        -r,--learningRate       (default 1e-3)          learning rate
        -l,--learningRateDecay  (default 1e-7)          learning rate decay
        -w,--weightDecay        (default 0)             weightDecay
        -m,--momentum           (default 0)             momentum
        -g,--epoch_step         (default 25)            epoch step
        -h,--model              (default overfeat)      model name
        -i,--max_epoch          (default 300)           maximum number of iterations
        -j,--backend            (default nn)            backend
]]

-- nb of threads and fixed seed (for repeatable experiments)
--if tensorType == 'float' then
--   print('==> switching to floats')
--   torch.setdefaulttensortype('torch.FloatTensor')
--elseif tensorType == 'cuda' then
--   print('==> switching to CUDA')
--   require 'cunn'
--   torch.setdefaulttensortype('torch.FloatTensor')
--end
--torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
local batchSize = opt.batchSize

opt.save = opt.save .. '_' .. dataset_code .. '_' .. experiment_code .. '_c' .. total_classes .. '_bs' .. opt.batchSize .. '_lr' .. opt.learningRate .. '_lrd' .. opt.learningRateDecay .. '_wd' .. opt.weightDecay .. '_m' .. opt.momentum .. '_' .. os.date("%y%m%d%H%M%S")
print(opt)
---------------------------------------------------------------------------------------------------------
print '==> construct network'

net = nn.Sequential()
local m = net.modules
if network == 'small' then
   print('==> init a small overfeat network')
   net:add(SpatialConvolutionMM(3, 96, 11, 11, 4, 4))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(96, 256, 5, 5, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(1024, 3072, 6, 6, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(3072, 4096, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(4096, total_classes, 1, 1, 1, 1))
   net:add(nn.View(total_classes))
   -- net:add(nn.ReLU(true))
   -- net:add(SpatialConvolutionMM(1000, total_classes, 1, 1, 1, 1))
   -- net:add(nn.View(total_classes))
   -- net:add(nn.SoftMax())
   net:add(nn.LogSoftMax())
   --print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weights')
   ParamBank:init("/home/cse/adarshc/Thesis/overfeat-weights/net_weight_0")
   ParamBank:read(        0, {96,3,11,11},    m[offset+1].weight)
   ParamBank:read(    34848, {96},            m[offset+1].bias)
   ParamBank:read(    34944, {256,96,5,5},    m[offset+4].weight)
   ParamBank:read(   649344, {256},           m[offset+4].bias)
   ParamBank:read(   649600, {512,256,3,3},   m[offset+7].weight)
   ParamBank:read(  1829248, {512},           m[offset+7].bias)
   ParamBank:read(  1829760, {1024,512,3,3},  m[offset+9].weight)
   ParamBank:read(  6548352, {1024},          m[offset+9].bias)
   ParamBank:read(  6549376, {1024,1024,3,3}, m[offset+11].weight)
   ParamBank:read( 15986560, {1024},          m[offset+11].bias)
   ParamBank:read( 15987584, {3072,1024,6,6}, m[offset+14].weight)
   ParamBank:read(129233792, {3072},          m[offset+14].bias)
   ParamBank:read(129236864, {4096,3072,1,1}, m[offset+16].weight)
   ParamBank:read(141819776, {4096},          m[offset+16].bias)
   -- ParamBank:read(141823872, {1000,4096,1,1}, m[offset+18].weight)
   -- ParamBank:read(145919872, {1000},          m[offset+18].bias)

elseif network == 'big' then
   print('==> init a big overfeat network')
   net:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolutionMM(96, 256, 7, 7, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(512, 512, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolutionMM(1024, 4096, 5, 5, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1))
   net:add(nn.ReLU(true))
   net:add(SpatialConvolutionMM(4096, total_classes, 1, 1, 1, 1))
   net:add(nn.View(total_classes))
   -- net:add(nn.ReLU(true))
   -- net:add(SpatialConvolutionMM(1000, total_classes, 1, 1, 1, 1))
   -- net:add(nn.View(total_classes))
   -- net:add(nn.SoftMax())
   net:add(nn.LogSoftMax())
   -- print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weights')
   ParamBank:init("/home/cse/adarshc/Thesis/overfeat-weights/net_weight_1")
   ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
   ParamBank:read(    14112, {96},            m[offset+1].bias)
   ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
   ParamBank:read(  1218432, {256},           m[offset+4].bias)
   ParamBank:read(  1218688, {512,256,3,3},   m[offset+7].weight)
   ParamBank:read(  2398336, {512},           m[offset+7].bias)
   ParamBank:read(  2398848, {512,512,3,3},   m[offset+9].weight)
   ParamBank:read(  4758144, {512},           m[offset+9].bias)
   ParamBank:read(  4758656, {1024,512,3,3},  m[offset+11].weight)
   ParamBank:read(  9477248, {1024},          m[offset+11].bias)
   ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+13].weight)
   ParamBank:read( 18915456, {1024},          m[offset+13].bias)
   ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+16].weight)
   ParamBank:read(123774080, {4096},          m[offset+16].bias)
   ParamBank:read(123778176, {4096,4096,1,1}, m[offset+18].weight)
   ParamBank:read(140555392, {4096},          m[offset+18].bias)
   -- ParamBank:read(140559488, {1000,4096,1,1}, m[offset+20].weight)
   -- ParamBank:read(144655488, {1000},          m[offset+20].bias)
end
-- close file pointer
ParamBank:close()

if cuda then net:cuda() end

-- net = torch.load('./VGG_FACE.t7')

print '==> here is the network:'
print(net)
---------------------------------------------------------------------------------
criterion = nn.ClassNLLCriterion()
if cuda then criterion:cuda() end
-- criterion = nn.CrossEntropyCriterion()
---------------------------------------------------------------------------------
print '==> loading dataset'

-- absolute path storing the training and test data directories
train_path = '/home/cse/adarshc/Thesis/FaceScrub/'
test_path = '/home/cse/adarshc/Thesis/FaceScrub/'

local train_files = require 'fs20_subset_train_images'
local test_files = require 'fs20_subset_test_images'
local train_labels = require 'fs20_subset_train_labels'
local test_labels = require 'fs20_subset_test_labels'

print '==> preparing training data'
trdata = {}
for i=1, trsize do
  xlua.progress(i, trsize)

  local img_raw = image.load(train_path .. train_files[i], 3):mul(255)
  -- local img_raw = image.load((train_path .. train_files[i]), 3, 'float'):mul(255)
  if img_raw:dim() == 2 then
    img_raw = img_raw:view(1,img_raw:size(1),img_raw:size(2))
  end
  if img_raw:size(1) == 1 then
    img_raw = img_raw:expand(3,img_raw:size(2),img_raw:size(3))
  end
  local img_dim
  if network == 'small' then    dim = 231
  elseif network == 'big' then  dim = 221 end
  local rh = img_raw:size(2)
  local rw = img_raw:size(3)
  if rh < rw then
    rw = math.floor(rw / rh * dim)
    rh = dim
  else
    rh = math.floor(rh / rw * dim)
    rw = dim
  end
  local img_scale = image.scale(img_raw, rw, rh)
  local offsetx = 1
  local offsety = 1
  if rh < rw then
    offsetx = offsetx + math.floor((rw-dim)/2)
  else
    offsety = offsety + math.floor((rh-dim)/2)
  end
  local img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]:floor()
  trdata[i] = img:add(-118.380948):div(61.896913)  -- fixed distn ~ N(118.380948, 61.896913^2)
end

trainData = {
   data = trdata,
   labels = train_labels,
   size = function() return trsize end
}

print '==> preparing testing data'
tedata = {}
for i=1, tesize do
  xlua.progress(i, tesize)

  local img_raw = image.load(test_path .. test_files[i], 3):mul(255)
  -- local img_raw = image.load((test_path .. test_files[i]), 3, 'float'):mul(255)
  if img_raw:dim() == 2 then
    img_raw = img_raw:view(1,img_raw:size(1),img_raw:size(2))
  end
  if img_raw:size(1) == 1 then
    img_raw = img_raw:expand(3,img_raw:size(2),img_raw:size(3))
  end
  local img_dim
  if network == 'small' then    dim = 231
  elseif network == 'big' then  dim = 221 end
  local rh = img_raw:size(2)
  local rw = img_raw:size(3)
  if rh < rw then
    rw = math.floor(rw / rh * dim)
    rh = dim
  else
    rh = math.floor(rh / rw * dim)
    rw = dim
  end
  local img_scale = image.scale(img_raw, rw, rh)
  local offsetx = 1
  local offsety = 1
  if rh < rw then
    offsetx = offsetx + math.floor((rw-dim)/2)
  else
    offsety = offsety + math.floor((rh-dim)/2)
  end
  local img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]:floor()
  tedata[i] = img:add(-118.380948):div(61.896913)  -- fixed distn ~ N(118.380948, 61.896913^2)
end

testData = {
   data = tedata,
   labels = test_labels,
   size = function() return tesize end
}

--print('[Caching similarity between samples...]')
--similarity_measures = torch.Tensor(trsize, trsize)
--for i = 1,trsize do
--  xlua.progress(i, trsize)
--  for j = 1,trsize do
--    similarity_measures[i][j] = torch.dist(trainData.data[i], trainData.data[j])
--  end
--end

untrained_training_images = {}
for i = 1,trsize do
  untrained_training_images[i] = true
end

-- initialising training set as 'initial_samples_per_class' images per class
--for i = 0,(total_classes - 1),1 do
--  for j = 1,initial_samples_per_class do
--    table.insert(labeled_pool_samples, trainData.data[i*samples_per_class + j])
--    table.insert(labeled_pool_labels, trainData.labels[i*samples_per_class + j])
--    untrained_training_images[i*samples_per_class + j] = false
--  end
--  current_pool_size = current_pool_size + initial_samples_per_class
--end
samples_required_per_class = {}
for j = 1,total_classes do
  samples_required_per_class[j] = initial_samples_per_class
end
for i = 1,trsize do
  if samples_required_per_class[trainData.labels[i]] > 0 then
    table.insert(labeled_pool_samples, trainData.data[i])
    table.insert(labeled_pool_labels, trainData.labels[i])
    untrained_training_images[i] = false
    current_pool_size = current_pool_size + 1
    samples_required_per_class[trainData.labels[i]] = samples_required_per_class[trainData.labels[i]] - 1
  end
end

local directory_name = opt.save
os.execute('mkdir -p ' .. directory_name)

local file_labeled_pool_size = io.open(paths.concat(opt.save, 'al_lps.txt'), 'a')
local file_selected_samples_list = io.open(paths.concat(opt.save, 'al_ssl.txt'), 'a')
local file_intermediate_classes_distribution = io.open(paths.concat(opt.save, 'al_icd.txt'), 'a')

file_labeled_pool_size:write(current_pool_size .. '\n')
file_selected_samples_list:write(table.concat(labeled_pool_labels, ",") .. '\n')

local intermediate_classes_distribution = {}
local tensor_labeled_pool_labels = torch.Tensor(labeled_pool_labels)
for j = 1,total_classes do
  local class_counter = 0
  for k = 1,current_pool_size do
    if tensor_labeled_pool_labels[k] == j then
      class_counter = class_counter + 1
    end
  end
  intermediate_classes_distribution[j] = class_counter
end
file_intermediate_classes_distribution:write(table.concat(intermediate_classes_distribution, ",") .. '\n')

file_labeled_pool_size:close()
file_selected_samples_list:close()
file_intermediate_classes_distribution:close()

-- print(labeled_pool_samples)
-- print(labeled_pool_labels)
----------------------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = torch.Tensor(total_classes)
for index = 1,total_classes do
  classes[index] = index
end

-- This matrix records the current confusion across classes
-- confusion = optim.ConfusionMatrix(target)
confusion = optim.ConfusionMatrix(label)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if net then
  parameters,gradParameters = net:getParameters()
end
----------------------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end
-----------------------------------------------------------------------------------------
--function calculateExpectedGradientLength(sample)
--  local input = sample
--  if tensorType == 'double' then input = input:double()
--  elseif tensorType == 'cuda' then input = input:cuda() end

--  local prediction, current_err, currentGradOutput, currentGradInput, current_module, current_input, top_k_predictions, top_k_classes, current_egl_weights, current_label
--  result = 0
--  prediction = net:forward(input)
--  top_k_predictions, top_k_classes = prediction:topk(egl_top_k_count, true)
--  top_k_predictions:apply(math.exp)
--  current_egl_weights = top_k_predictions:div(top_k_predictions:sum())

--  current_parameters,current_grad_parameters = net.modules[20]:getParameters()
--  --for label = 1,total_classes do
--  for counter = 1,egl_top_k_count do
--    current_label = top_k_classes[counter]
--    net:zeroGradParameters()

--    current_err = criterion:forward(prediction, current_label)
--    currentGradOutput = criterion:backward(prediction, current_label)

--    current_module = net.modules[22]
--    current_input = net:get(21).output
--    currentGradInput = current_module:backward(current_input, currentGradOutput)
--    currentGradOutput = currentGradInput

--    current_module = net.modules[21]
--    current_input = net:get(20).output
--    currentGradInput = current_module:backward(current_input, currentGradOutput)
--    currentGradOutput = currentGradInput

--    current_module = net.modules[20]
--    current_input = net:get(19).output
--    currentGradInput = current_module:backward(current_input, currentGradOutput)

--    result = result + current_egl_weights[counter]*torch.norm(current_grad_parameters)
--  end

--  return result
--end
-----------------------------------------------------------------------------------------
function calculateDensityWeightedExpectedModelChange()
  result = torch.Tensor((trsize - current_pool_size))
  unlabeled_training_indices = torch.Tensor((trsize - current_pool_size))

  print('[Computing feature vectors and expected model changes of unlabeled pool samples...]')
  local points = torch.Tensor((trsize - current_pool_size), 4096)
  local dw_counter, egl_counter, input, prediction, current_output, current_err, currentGradOutput, currentGradInput, current_module, current_input, top_k_predictions, top_k_classes, current_egl_weights, current_label
  dw_counter = 0
  for i = 1,trsize do
    xlua.progress(i, trsize)
    if untrained_training_images[i]  then
      dw_counter = dw_counter + 1
      input = trainData.data[i]
      if tensorType == 'double' then input = input:double()
      elseif tensorType == 'cuda' then input = input:cuda() end
      prediction = net:forward(input)
      current_output = net:get(19).output
      
      points[dw_counter] = current_output:clone():float()
      unlabeled_training_indices[dw_counter] = i

      --local probability_distribution = prediction:clone()
      --probability_distribution:apply(math.exp)
      --result[dw_counter] = torch.max(probability_distribution)

      result[dw_counter] = 0
      local top_k_predictions, top_k_classes = prediction:topk(egl_top_k_count, true)
      top_k_predictions:apply(math.exp)
      local current_egl_weights = top_k_predictions:div(top_k_predictions:sum())

      current_parameters,current_grad_parameters = net.modules[20]:getParameters()
      for egl_counter = 1,egl_top_k_count do
        current_label = top_k_classes[egl_counter]
        net:zeroGradParameters()

        current_err = criterion:forward(prediction, current_label)
        currentGradOutput = criterion:backward(prediction, current_label)

        current_module = net.modules[22]
        current_input = net:get(21).output
        currentGradInput = current_module:backward(current_input, currentGradOutput)
        currentGradOutput = currentGradInput

        current_module = net.modules[21]
        current_input = net:get(20).output
        currentGradInput = current_module:backward(current_input, currentGradOutput)
        currentGradOutput = currentGradInput

        current_module = net.modules[20]
        current_input = net:get(19).output
        currentGradInput = current_module:backward(current_input, currentGradOutput)

        result[dw_counter] = result[dw_counter] + current_egl_weights[egl_counter]*torch.norm(current_grad_parameters)
      end
    end
  end
  --print(result)

  print('[Clustering unlabeled pool features...]')
  local centroids, counts
  centroids, counts = unsup.kmeans(points, total_classes, k_means_iterations, total_classes, function(i, _, totalcounts) if i < k_means_iterations then totalcounts:zero() end end, true)
  --counts:div(k_means_iterations)
  --print(counts)
  --print(counts:sum())

  print('[Calculating information density measures...]')
  local information_density_measures = torch.Tensor((trsize - current_pool_size))
  for i = 1,(trsize - current_pool_size) do
    xlua.progress(i, (trsize - current_pool_size))
    local aggregate_similarity_measure = 0
    for j = 1,total_classes do
      aggregate_similarity_measure = aggregate_similarity_measure + (torch.dist(points[i], centroids[j]) * counts[j])
    end
    information_density_measures[i] = aggregate_similarity_measure
  end

  information_density_measures:div(counts:sum())
  --print(information_density_measures)
  result:cdiv(information_density_measures)
  --print(result)

  return result, unlabeled_training_indices
end
-----------------------------------------------------------------------------------------
print '==> defining training procedure'

function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   local correct_predictions = 0

    -- check whether active learning needs to be performed for this epoch
    if current_pool_size >= training_threshold then
      active_learning_flag = false
    elseif epoch_counter < epoch_per_update then
      active_learning_flag = false
      epoch_counter = epoch_counter + 1
    elseif initial_training_counter > 0 then
      active_learning_flag = false
    else
      active_learning_flag = true
      epoch_counter = 0
    end

    -- add the most informative sample to training data
    -- if active_learning_flag then
    --   for i = 0,(total_classes - 1) do
    --     local most_informative_measure = 0
    --     local most_informative_index = (i*samples_per_class + 1)
    --     for j = (i*samples_per_class + 1),((i + 1)*samples_per_class) do
    --       if untrained_training_images[j] then
    --         local current_measure = calculateConfidence(trainData.data[j])
    --         if most_informative_measure < current_measure then
    --           most_informative_measure = current_measure
    --           most_informative_index = j
    --         end
    --       end
    --     end
    --     table.insert(labeled_pool_samples, trainData.data[most_informative_index])
    --     table.insert(labeled_pool_labels, trainData.labels[most_informative_index])
    --     untrained_training_images[most_informative_index] = false
    --   end
    -- end
    if active_learning_flag then
      print '==> active learning: ACTIVE...'
      print '==> updating the labeled pool:'
      net:evaluate()

      --EGL_measures = torch.Tensor(trsize)
      --for i = 1,trsize do
      --  xlua.progress(i, trsize)
      --  if untrained_training_images[i] then
      --    EGL_measures[i] = calculateExpectedGradientLength(trainData.data[i])
      --  else
      --    EGL_measures[i] = 0
      --  end
      --end

      density_weighted_emc_measures, unlabeled_training_indices = calculateDensityWeightedExpectedModelChange()

      top_k_values, top_k_indices = density_weighted_emc_measures:topk(samples_added_per_update, true)
      for i = 1,samples_added_per_update do
        --if untrained_training_images[top_k_indices[i]] then
        table.insert(labeled_pool_samples, trainData.data[unlabeled_training_indices[top_k_indices[i]]])
        table.insert(labeled_pool_labels, trainData.labels[unlabeled_training_indices[top_k_indices[i]]])
        current_pool_size = current_pool_size + 1
        untrained_training_images[unlabeled_training_indices[top_k_indices[i]]] = false
        --end
      end
    end

    print("==> current labeled pool: ", table.concat(labeled_pool_labels, ", "))
    print("==> current pool size: " .. current_pool_size)
    if active_learning_flag then
      print("[saving active learning statistics...]")
      local file_labeled_pool_size = io.open(paths.concat(opt.save, 'al_lps.txt'), 'a')
      local file_selected_samples_list = io.open(paths.concat(opt.save, 'al_ssl.txt'), 'a')
      local file_intermediate_classes_distribution = io.open(paths.concat(opt.save, 'al_icd.txt'), 'a')

      file_labeled_pool_size:write(current_pool_size .. '\n')
      file_selected_samples_list:write(table.concat(labeled_pool_labels, ",") .. '\n')

      local intermediate_classes_distribution = {}
      local tensor_labeled_pool_labels = torch.Tensor(labeled_pool_labels)
      for j = 1,total_classes do
        local class_counter = 0
        for k = 1,current_pool_size do
          if tensor_labeled_pool_labels[k] == j then
            class_counter = class_counter + 1
          end
        end
        intermediate_classes_distribution[j] = class_counter
      end
      file_intermediate_classes_distribution:write(table.concat(intermediate_classes_distribution, ",") .. '\n')

      file_labeled_pool_size:close()
      file_selected_samples_list:close()
      file_intermediate_classes_distribution:close()

      local directory_name = paths.concat(opt.save, 'actively_selected_samples')
      os.execute('mkdir -p ' .. directory_name)
      for counter = 1,current_pool_size do
        image.save(paths.concat(directory_name, 'sample_' .. counter .. '.jpg'), labeled_pool_samples[counter])
      end
    end

   -- shuffle at each epoch
   shuffle = torch.randperm(current_pool_size)

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   net:training()

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,current_pool_size,batchSize do
      -- disp progress
      xlua.progress(t, current_pool_size)

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,current_pool_size) do
         -- load new sample
         -- local input = trainData.data[shuffle[i]]
         local input = labeled_pool_samples[shuffle[i]]
         -- local target = trainData.labels[shuffle[i]]
         local target = labeled_pool_labels[shuffle[i]]
         if tensorType == 'double' then input = input:double()
         elseif tensorType == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = net:forward(inputs[i])
                          prob, idx = torch.max(output, 1)
                          if idx[1] == targets[i] then
                            correct_predictions = correct_predictions + 1
                          end
                          --output = output:float()
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          net:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                          -- confusion:add(output, label[targets[i]])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / current_pool_size
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   --if opt.plot then
   --   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   --   trainLogger:plot()
   --end

   -- save/log current net
   if epoch % 10 == 0 then
     local filename = paths.concat(opt.save, 'net.net')
     os.execute('mkdir -p ' .. sys.dirname(filename))
     print('==> saving model to '.. filename)
     torch.save(filename, net)
   end

   -- next epoch
   confusion:zero()
   epoch = epoch + 1

   print('==> class accuracy while TRAINING: ' .. correct_predictions .. ' out of ' .. current_pool_size)
end
---------------------------------------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   net:evaluate()

   local correct_predictions = 0

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if tensorType == 'double' then input = input:double()
      elseif tensorType == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = net:forward(input)
      prob, idx = torch.max(pred, 1)
      if idx[1] == target then
        correct_predictions = correct_predictions + 1
      end
      -- print('==> sample #' .. t .. ':')
      -- print(target)
      -- print(idx)
      -- print(label[idx:squeeze()], prob:squeeze())

      confusion:add(pred, target)
      -- confusion:add(output, label[targets])
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   --if opt.plot then
   --   testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --   testLogger:plot()
   --end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()

   print('==> class accuracy while TESTING: ' .. correct_predictions .. ' out of ' .. tesize)
end
-----------------------------------------------------------------------------------------------------------------------
-- ==> here is the network:  
-- nn.Sequential {
--   [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> output]
--   (1): nn.SpatialConvolution(3 -> 96, 7x7, 2,2)
--   (2): nn.ReLU
--   (3): nn.SpatialMaxPooling(3,3,3,3)
--   (4): nn.SpatialConvolutionMM(96 -> 256, 7x7)
--   (5): nn.ReLU
--   (6): nn.SpatialMaxPooling(2,2,2,2)
--   (7): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
--   (8): nn.ReLU
--   (9): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
--   (10): nn.ReLU
--   (11): nn.SpatialConvolutionMM(512 -> 1024, 3x3, 1,1, 1,1)
--   (12): nn.ReLU
--   (13): nn.SpatialConvolutionMM(1024 -> 1024, 3x3, 1,1, 1,1)
--   (14): nn.ReLU
--   (15): nn.SpatialMaxPooling(3,3,3,3)
--   (16): nn.SpatialConvolutionMM(1024 -> 4096, 5x5)
--   (17): nn.ReLU
--   (18): nn.SpatialConvolutionMM(4096 -> 4096, 1x1)
--   (19): nn.ReLU
--   (20): nn.SpatialConvolutionMM(4096 -> 20, 1x1)
--   (21): nn.View(20)
--   (22): nn.LogSoftMax
-- }
print '==> the main loop'
print '==> the main loop'
while true do
  if initial_training_counter > 0 then
    train()
    initial_training_counter = initial_training_counter - 1
  else
    --if current_pool_size <= (0.25*training_threshold) then
    --  epoch_per_update = 3
    --elseif current_pool_size <= (0.5*training_threshold) then
    --  epoch_per_update = 4
    --else
    --  epoch_per_update = 5
    --end
    print('[training using density weighted expected gradient length method...]')
    train()
    print('[testing using density weighted expected gradient length method...]')
    test()
  end
end
--print(net.modules[20])
--net:evaluate()
--pred = net:forward(trainData.data[1]:cuda())
--print(pred)
--sum = 0
--for i = 1,total_classes do
--  sum = sum + math.exp(pred[i])
--end
--print('==> sum of predicted values for each class: ' .. sum)
--sample_result, sample_indices = calculateDensityWeightedExpectedModelChange()
--print(sample_result)
--for i = 81,2000 do
--  current_measure = calculateExpectedGradientLength(trainData.data[i])
--  --print('==> current EGL measure: ' .. current_measure)
--  print(current_measure)
--end
--current_measure = torch.Tensor(100)
--for i = 1,100 do
--  xlua.progress(i, 100)
--  current_measure[i] = calculateExpectedGradientLength(trainData.data[i])
--end
-----------------------------------------------------------------------------------------------------------------------
