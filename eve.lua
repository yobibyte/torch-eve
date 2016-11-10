-- adam code taken from https://raw.githubusercontent.com/torch/optim/master/adam.lua

--[[n implementation of Adam http://arxiv.org/pdf/1412.6980.pdf

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- `config.learningRateDecay` : learning rate decay
- 'config.beta1'             : first moment coefficient
- 'config.beta2'             : second moment coefficient
- 'config.beta3'             : exponential decay rate for computing relative change
- 'config.epsilon'           : for numerical stability
- 'config.weightDecay'       : weight decay
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
- 'k'                        : lower threshold for relative change
- 'K'                        : upper threshold for relative change

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function optim.adam(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 0.001
   local lrd = config.learningRateDecay or 0

   local beta1 = config.beta1 or 0.9
   local beta2 = config.beta2 or 0.999
   local beta3 = config.beta3 or 0.999
   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0

   local k_lower = config.k_lower or 0.1
   local k_upper = config.k_upper or 10


   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)

   -- (2) weight decay
   if wd ~= 0 then
      dfdx:add(wd, x)
   end

   -- Initialization
   state.t = state.t or 0
   -- Exponential moving average of gradient values
   state.m = state.m or x.new(dfdx:size()):zero()
   -- Exponential moving average of squared gradient values
   state.v = state.v or x.new(dfdx:size()):zero()
   -- A tmp tensor to hold the sqrt(v) + epsilon
   state.denom = state.denom or x.new(dfdx:size()):zero()
   -- relative change moving average
   state.d = state.d or 1
   -- objective
   state.f_prev = state.f_prev or x.new(dfdx:size()):zero()
   -- objective
   state.f_preprevv = state.f_preprev or x.new(dfdx:size()):zero()


   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + state.t*lrd)

   state.t = state.t + 1

   -- Decay the first and second moment running average coefficient
   state.m:mul(beta1):add(1-beta1, dfdx)
   state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

   if state.t > 1 then

      local delta_lower = k_lower+1
      local delta_upper = k+upper+1
      if fx <= state.f then
         delta_lower = 1/(k_lower+1)
         delta_upper = 1/(k_upper+1)         
      local c = min(max(delta_lower, fx/state.f_preprev), delta_upper)  
      local tmp = state.f_prev
      state.f_prev = c*state.f_preprev
      state.f_preprev = tmp
      local ri
      r:copy(state.f_prev):csub(state.f_preprev):abs()/torch.cmin(state.f_prev, state.f_preprev)
      state.d:mul(beta3):add(1-beta3):mul(r)
      local biasCorrection1 = 1 - beta1^state.t
      local biasCorrection2 = 1 - beta2^state.t
      local stepSize = clr * math.sqrt(biasCorrection2)/biasCorrection1
   else:
      state.f_preprev = state.f_prev
      state.f_prev    = fx 
      state.d = 1

   state.denom:copy(state.d:mul(state.v:sqrt()):add(epsilon))
   -- (4) update x
   x:addcdiv(-stepSize, state.m, state.denom)

   -- return x*, f(x) before optimization
   return x, {fx}
end
