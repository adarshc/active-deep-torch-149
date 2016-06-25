-- final_7.lua
------------------------------------------------------------------------------------------------------
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
------------------------------------------------------------------------------------------------------
-- global parameters
local ParamBank = require 'ParamBank'
local label     = require 'sc_subset_classes'
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

dataset_code = 'sc'
experiment_code = 'rs'
total_classes = 196
-- training/test size
trsize = 8144
tesize = 8041
-- type of OverFeat conv-net
local network  = 'big'
-- local batchSize = 8
-- flag denoting when to begin active-learning
-- active_learning_flag = false
-- epoch_per_update = 5
-- epoch_counter = 0
-- initial_training_counter = 20
-- samples_per_class = trsize / total_classes -- depends on the configuration of the training metadata
-- samples_added_per_update = 10
optimization_factor = 0.25
training_threshold = 2072
initial_samples_per_class = 2
labeled_pool_samples = {}
labeled_pool_labels = {}
current_pool_size = 0

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
--cmd:option('-plot', true, 'live plot')
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
train_path = '/home/cse/adarshc/Thesis/stanford_cars/'
test_path = '/home/cse/adarshc/Thesis/stanford_cars/'

local train_files = require 'sc_subset_train_images'
local test_files = require 'sc_subset_test_images'
local train_labels = require 'sc_subset_train_labels'
local test_labels = require 'sc_subset_test_labels'

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

untrained_training_images = {}
for i = 1,trsize do
  untrained_training_images[i] = true
end

-- initialising training set as 'initial_samples_per_class' images per class
-- for i = 0,(total_classes - 1),1 do
--   table.insert(labeled_pool_samples, trainData.data[i*samples_per_class + 1])
--   table.insert(labeled_pool_labels, trainData.labels[i*samples_per_class + 1])
--   table.insert(labeled_pool_samples, trainData.data[i*samples_per_class + 2])
--   table.insert(labeled_pool_labels, trainData.labels[i*samples_per_class + 2])
--   untrained_training_images[i*samples_per_class + 1] = false
--   untrained_training_images[i*samples_per_class + 2] = false
--   current_pool_size = current_pool_size + 2
-- end
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

-- randomly selecting remaining (training_threshold - (initial_samples_per_class*total_classes)) samples from the unlabeled pool
randomize = torch.randperm(trsize)
for i = 1,trsize do
  if untrained_training_images[randomize[i]] then
    table.insert(labeled_pool_samples, trainData.data[randomize[i]])
    table.insert(labeled_pool_labels, trainData.labels[randomize[i]])
    untrained_training_images[randomize[i]] = false
    current_pool_size = current_pool_size + 1
    if current_pool_size >= training_threshold then
      break
    end
  end
end

-- print(labeled_pool_samples)
-- print(labeled_pool_labels)

for counter = 1,current_pool_size do
  image.save('./actively_selected_samples/sample_' .. counter .. '.jpg', labeled_pool_samples[counter])
end
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
-- function calculateEntropy(sample)
--   -- net:evaluate()
--   local prediction = net:forward(sample)
--   local normalized_output = torch.Tensor(total_classes)
--   for i = 1,total_classes do
--     normalized_output[i] = math.exp(prediction[i])
--   end
--   local max_prob = torch.max(normalized_output, 1)
--   result = torch.add(-max_prob, 1.0)
--   -- net:training()
--   return result
-- end
-----------------------------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   local correct_predictions = 0

    -- -- check whether active learning needs to be performed for this epoch
    --if current_pool_size >= training_threshold then
    --  active_learning_flag = false
    --elseif epoch_counter < epoch_per_update then
    --  active_learning_flag = false
    --  epoch_counter = epoch_counter + 1
    --elseif initial_training_counter > 0 then
    --  active_learning_flag = false
    --else
    --  active_learning_flag = true
    --  epoch_counter = 0
    --end

    -- -- add the most informative sample to training data
    -- if active_learning_flag then
    --   for i = 0,(total_classes - 1) do
    --     local most_informative_measure = 0
    --     local most_informative_index = (i*samples_per_class + 1)
    --     for j = (i*samples_per_class + 1),((i + 1)*samples_per_class) do
    --       if untrained_training_images[j] then
    --         local current_measure = calculateEntropy(trainData.data[j])
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
    -- -- if active_learning_flag then
    --   print '==> active learning: ACTIVE...'
    --   print '==> updating the labeled pool:'
    --   entropy_measures = torch.Tensor(trsize)
    --   for i = 1,trsize do
    --     xlua.progress(i, trsize)
    --     if untrained_training_images[i] then
    --       entropy_measures[i] = calculateEntropy(trainData.data[i])[1]
    --     else
    --       entropy_measures[i] = 0
    --     end
    --   end
    --   top_k_values, top_k_indices = entropy_measures:topk(samples_added_per_update, true)
    --   for i = 1,samples_added_per_update do
    --     if untrained_training_images[top_k_indices[i]] then
    --       table.insert(labeled_pool_samples, trainData.data[top_k_indices[i]])
    --       table.insert(labeled_pool_labels, trainData.labels[top_k_indices[i]])
    --       current_pool_size = current_pool_size + 1
    --       untrained_training_images[top_k_indices[i]] = false
    --     end
    --   end
    -- end

    print("==> training on fixed labeled pool: ", table.concat(labeled_pool_labels, ", "))
    print("==> labeled pool size: " .. current_pool_size)

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
print '==> the main loop'
--while true do
--  if initial_training_counter > 0 then
--    train()
--    initial_training_counter = initial_training_counter - 1
--  else
--    if current_pool_size <= (0.25*training_threshold) then
--      epoch_per_update = 3
--    elseif current_pool_size <= (0.5*training_threshold) then
--      epoch_per_update = 4
--    else
--      epoch_per_update = 5
--    end
--    train()
--    test()
--  end
--end
while true do
  train()
  test()
end
-----------------------------------------------------------------------------------------------------------------------
