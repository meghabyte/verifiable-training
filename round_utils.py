
"""
General utilities for handling verification via pytorch hooks.
"""
import torch

THRESHOLDS = {}
"""
Thresholds for adaptive thresholding procedure, for example:

THRESHOLDS = {"conv":0.295, "bn":0.4}
"""


def check_bits(x, target_int, mask_int, precision_type):
    if(x.dtype != precision_type):
        print("Mismatch precision types")
        return None
    elif(precision_type==torch.float32):
        mask_tensor = torch.tensor(mask_int, dtype=torch.int32)
        target_tensor = torch.tensor(target_int, dtype=torch.int32)
        z = x.view(torch.int)
        result = z.bitwise_and(mask_tensor)
        del mask_tensor
        del z
        result_bool = (result == target_tensor)
        del target_tensor
        return result_bool
    elif(precision_type==torch.float64):
        mask_tensor = torch.tensor(mask_int, dtype=torch.int64)
        target_tensor = torch.tensor(target_int, dtype=torch.int64)
        z = x.view(torch.long)
        result = z.bitwise_and(mask_tensor)
        del mask_tensor
        del z
        result_bool = (result == target_tensor)
        del target_tensor
        return result_bool
    return None

def apply_nextafter_n(x, n=1, direction=torch.inf, precision=torch.float32):
    next_x = x
    direction_tensor = torch.tensor([direction], dtype=precision, device=x.device)
    for i in range(n):
        new_next_x = NextAfter.apply(next_x, direction_tensor) 
        del next_x
        next_x = new_next_x
    if(torch.numel(next_x) == 1):
        return torch.squeeze(next_x)
    del direction_tensor
    return next_x


class NextAfter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, direction):
        #ctx.mark_dirty(input)
        return torch.nextafter(input, direction)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
class Round64Down(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rounded_input, rounder):
        return rounder.round_val_64_down(input, rounded_input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
    
class Round64Up(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rounded_input, rounder):
        #ctx.mark_dirty(input)
        return rounder.round_val_64_up(input, rounded_input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class RoundStandard(torch.autograd.Function):   
    @staticmethod
    def forward(ctx, input, round_amt):
        return round_any_final(input, round_bit=round_amt)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  

class Rounder:
    def __init__(self, round_amount=32):
        self.round_amount = round_amount
    
    def apply_round(self, x):
        if(self.round_amount not in range(23, 35)):
            print("Invalid round amount, rounding to 32 bit instead.")
            return round_32(x)
        elif (self.round_amount == 32):
            return round_32(x)
        elif (self.round_amount == 16):
            return x.to(torch.float16).to(torch.float64)
        else:
            return RoundStandard.apply(x, self.round_amount)
        
    def round_val_64_down(self, x, rounded_x):
        degree_move_float = 2**(32-self.round_amount)
        return apply_nextafter_n(rounded_x.to(torch.float32), n=degree_move_float, direction=-torch.inf).to(dtype=torch.float64)
    
    def round_val_64_up(self, x, rounded_x):
        degree_move_float = 2**(32-self.round_amount)
        return apply_nextafter_n(rounded_x.to(torch.float32), n=degree_move_float, direction=torch.inf).to(dtype=torch.float64)
    
    def get_logging_condition_counts(self, v, round_v, name):
        #if((torch.pow(10, torch.floor(torch.log10(torch.abs(v)))) != torch.pow(10, torch.floor(torch.log10(torch.abs(round_v))))).any()):
            #print("EXP MISMATCH \n")
        if(self.round_amount == 32):
            eps = 2**(-23)
        else:
            diff_bits = 32-self.round_amount
            eps =  (2**diff_bits) * (2**(-23))
        exp = torch.pow(10, torch.floor(torch.log10(torch.abs(v))))
        pivot = 0.25 # modify this if using adaptive thresholding procedure (use module name to identify layer)
        threshold = exp * torch.tensor([0.5*eps - ((0.5-pivot)*(2**(-23)))], dtype=torch.float64, device='cuda:0')
        diff = torch.abs(v - round_v)
        result = diff > threshold
        return result.sum(), (torch.abs(v)>=0).sum()
    
    def check_logging_threshold(self, v, round_v, name):
        #if((torch.pow(10, torch.floor(torch.log10(torch.abs(v)))) != torch.pow(10, torch.floor(torch.log10(torch.abs(round_v))))).any()):
            #print("EXP MISMATCH \n")
        if(self.round_amount == 32):
            eps = 2**(-23)
        else:
            diff_bits = 32-self.round_amount
            eps =  (2**diff_bits) * (2**(-23))
        exp = torch.pow(10, torch.floor(torch.log10(torch.abs(v))))
        pivot = 0.25 # modify this if using adaptive thresholding procedure (use module name to identify layer)
        threshold = exp * torch.tensor([0.5*eps - ((0.5-pivot)*(2**(-23)))],  dtype=torch.float64, device='cuda:0')
        diff = torch.abs(v - round_v)
        result = diff > threshold
        del threshold
        del diff
        return result

def round_32(x):
    #Round float64 to float32
    rounded_x = x.to(torch.float32)
    return rounded_x.to(torch.float64)

def print_memory_usage():
  stats = torch.cuda.memory_stats(device='cuda:'+str(torch.cuda.current_device()))
  print(f"allocated: {stats['allocation.all.current']} {stats['allocated_bytes.all.current']} | active: {stats['active.all.current']} {stats['active_bytes.all.current']}")
    

def round_any_final(tensor, round_bit=30, debug=False, dtype=torch.float64):
  #print_memory_usage()
  if not tensor.dtype == torch.float64:
    raise ValueError("Input must be of type torch.float64 but is of type {tensor.dtype}.")
  x = torch.where(tensor >= 0, tensor, -tensor)
  x_float32 = x.to(torch.float32)
  if round_bit==32:
    return x_float32.to(dtype)
  k = 32 - round_bit # of lower bits to retain
  x_category = x_float32.view(torch.int)%(2**k)
  x_round_needed = (x_category != 0)
  x_halfway = (x_category == 2**(k-1))
  x_round_up   = torch.logical_or(torch.logical_and(x_round_needed, x_category > 2**(k-1)), torch.logical_and(x_halfway,(x >= x_float32)))
  x_round_up_count = torch.where(x_round_up, 2**k - x_category, torch.zeros_like(x_round_up, dtype=torch.int32))
  x_round_down = torch.logical_or(torch.logical_and(x_round_needed, x_category < 2**(k-1)), torch.logical_and(x_halfway,(x < x_float32)))
  x_round_down_count = torch.where(x_round_down, x_category, torch.zeros_like(x_round_down, dtype=torch.int32))
  if debug:
    print("x >= x_float32:", x >= x_float32)
    print(f"category={x_category}")
    print(f"upcount={x_round_up_count}")
    print(f"downcount={x_round_down_count}")
  x_rounded = x_float32
  x_like_inf = torch.full_like(x_rounded, torch.inf)
  for i in range(1,1+2**(k-1)):
    if debug:
      print(f"\ni={i}")
    x_rounded_nextafter = NextAfter.apply(x_rounded,x_like_inf)
    if debug:
      print(x_rounded_nextafter-x_rounded)
    x_rounded = torch.where(x_round_up_count>0,x_rounded_nextafter,x_rounded)
    x_round_up_count = torch.where(x_round_up_count>0,x_round_up_count-1,x_round_up_count)
    if debug:
      print(f"upcount={x_round_up_count}")
    x_rounded_justbefore = NextAfter.apply(x_rounded,-x_like_inf)
    if debug:
      print(x_rounded-x_rounded_justbefore)
    x_rounded = torch.where(x_round_down_count>0,x_rounded_justbefore,x_rounded)
    x_round_down_count = torch.where(x_round_down_count>0,x_round_down_count-1,x_round_down_count)
    if debug:
      print(f"downcount={x_round_down_count}")
  x_rounded = torch.where(tensor>=0, x_rounded, -x_rounded)
  if dtype == torch.float32:
    return x_rounded
  else:
    return x_rounded.to(dtype)

def float_to_binary_32(num):
    neg = (num < 0)
    if(neg):
        num = num*-1
    if not num.dtype == torch.float32:
        raise ValueError("Input must be of type torch.float32.")
    # Convert the float to its binary representation
    binary = bin(int(num.view(dtype=torch.int32).numpy()))[2:]
    # Pad the binary representation with leading zeros if necessary
    binary = binary.zfill(32)
    if(neg):
        binary = "1"+binary[1:]

    return binary

def float_to_binary_64(num):
    neg = (num < 0)
    if(neg):
        num = num*-1
    if not num.dtype == torch.float64:
        raise ValueError("Input must be of type torch.float32.")
    # Convert the float to its binary representation
    binary = bin(int(num.view(dtype=torch.int64).numpy()))[2:]
    # Pad the binary representation with leading zeros if necessary
    binary = binary.zfill(64)
    if(neg):
        binary = "1"+binary[1:]

    return binary