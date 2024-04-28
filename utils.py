"""
General utilities for handling verification via pytorch hooks.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import hashlib
from functools import partial
from round_utils import *
from logging_utils import *
m = hashlib.sha256()


class CustomDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, training=True):
        ctx.save_for_backward(input, mask)
        ctx.training = training

        if ctx.training:
            output = input * mask
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        training = ctx.training

        if training:
            grad_input = grad_output * mask
        else:
            grad_input = grad_output

        return grad_input, None, None


def set_model_type(model, model_type=torch.float32):
        if(model_type == torch.float16):
            convert_to_half_precision(model)
        else:
            model.to(model_type)

def convert_to_half_precision(model):
        model.to(torch.float16)
        # make batch norm layers float32 for convergence issues
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.to(torch.float32)
                
                
def disable_inplace_activation(model):
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.ReLU):
                # Replace ReLU activation with a new ReLU activation with inplace=False
                setattr(model, child_name, torch.nn.ReLU(inplace=False))
            else:
                disable_inplace_activation(child)

class HookCreator:
    def __init__(self, check_logs=None, rounding=False, do_log=False, forward_log_fn="", backward_log_fn="", back_audit=None, forward_audit=None, round_amount=32):
        self.rounding = rounding
        self.do_log = do_log
        self.curr_hook_logs = []
        self.round_amount = round_amount
        self.rounder = Rounder(self.round_amount)

        self.forward_log_fn = forward_log_fn
        self.forward_log = forward_log_fn
        self.backward_log_fn = backward_log_fn
        self.backward_log = backward_log_fn
        self.forward_logger = None
        self.backward_logger = None
        
        # Audit logs
        self.back_audit_fn = None
        self.forward_audit_fn = None
        self.back_audit = None
        self.forward_audit = None
        self.forward_audit_reader = None
        self.back_audit_reader = None
        
        # GPT2 Dropout
        self.GPT2Dropout = torch.nn.Dropout(0.1)
        self.DropoutP = 0.1
        self.dropout_seed = 0
        self.gcuda = torch.Generator(device='cuda')
        
        if(back_audit):
            self.back_audit_fn = back_audit
            self.back_audit = self.back_audit_fn
        if(forward_audit):
            self.forward_audit_fn = forward_audit
            self.forward_audit = self.forward_audit_fn

        self.threshold_counters = [0, 0]
        
    def save_rounding_logs(self):
        del self.forward_logger
        del self.backward_logger
        
    def del_audit_readers(self):
        if(self.forward_audit_reader):
            del self.forward_audit_reader
        if(self.back_audit_reader):
            del self.back_audit_reader
        
    def start_rounding_logs(self):
        self.forward_logger = Tensor012Logger(self.forward_log, threshold=-1)
        self.backward_logger = Tensor012Logger(self.backward_log, threshold=-1)
        
    def start_audit_readers(self):
        if(self.forward_audit):
            print("started forward reader: "+self.forward_audit)
            self.forward_audit_reader = Tensor012Logger(self.forward_audit, mode="rb")
        if(self.back_audit):
            print("started backward reader: "+self.back_audit)
            self.back_audit_reader = Tensor012Logger(self.back_audit, mode="rb")
        
    def hook_creator_update_fns(self, batch_fn=0, epoch_fn=0):
        self.forward_log = self.forward_log_fn+"_b_"+str(batch_fn)+"_e_"+str(epoch_fn)
        self.backward_log = self.backward_log_fn+"_b_"+str(batch_fn)+"_e_"+str(epoch_fn)
        if(self.forward_audit):
            self.forward_audit = self.forward_audit_fn+"_b_"+str(batch_fn)+"_e_"+str(epoch_fn)
        if(self.back_audit):
            self.back_audit = self.back_audit_fn+"_b_"+str(batch_fn)+"_e_"+str(epoch_fn)

    def update_threshold_counters(self, threshold_counters):
        for i in range(len(self.threshold_counters)):
            self.threshold_counters[i] = self.threshold_counters[i] + threshold_counters[i]


            
    def rounding_backward_hook_64(self, name, grad):
        if("transformer" in name):
            torch.manual_seed(self.dropout_seed)
            self.dropout_seed += 1
        rounded_log=None
        if self.rounding and self.back_audit:
            new_grad, rounded_log, threshold_counters = round_val_64_to_32(grad, self.rounder, use_log=self.back_audit_reader, name=name)
            self.update_threshold_counters(threshold_counters)
        elif self.rounding:
            new_grad, rounded_log, threshold_counters = round_val_64_to_32(grad, self.rounder)
            self.update_threshold_counters(threshold_counters)
        else: 
            new_grad = grad
        if self.do_log:
            self.backward_logger.put_tensor(rounded_log)
        return new_grad


    def full_rounding_backward_hook_64(self, name, module, old_grad_input, old_grad_output):
        grad_input = old_grad_input
        if("transformer" in name and "drop" in name):
            # sepcial handling  nondeterminism in dropout
            torch.manual_seed(self.dropout_seed)
            np.random.seed(self.dropout_seed)
            random.seed(self.dropout_seed)
            grad_input = old_grad_output #override
        if("transformer" in name):
            torch.manual_seed(self.dropout_seed)
            self.dropout_seed += 1
        rounded_log=None
        new_grad_tuple = []
        for grad in grad_input:
            if grad is None:
                new_grad_tuple.append(grad)
                continue
            if self.rounding and self.back_audit:
                new_grad, rounded_log, threshold_counters = round_val_64_to_32(grad.clone(), self.rounder, use_log=self.back_audit_reader, name=name)
                self.update_threshold_counters(threshold_counters)
            elif self.rounding:
                new_grad, rounded_log, threshold_counters = round_val_64_to_32(grad.clone(), self.rounder, name=name)
                self.update_threshold_counters(threshold_counters)
            else:
                new_grad = grad.clone()
            new_grad_tuple.append(new_grad)
            if self.do_log:
                self.backward_logger.put_tensor(rounded_log)
        return tuple(new_grad_tuple)

    def get_random_int_str(self, match_shape):
        torch.manual_seed(self.dropout_seed)
        np.random.seed(self.dropout_seed)
        random.seed(self.dropout_seed)
        return_int_str = torch.zeros(match_shape.shape, device="cuda:0")
        for i in range(match_shape.shape[2]):
            return_int_str[:, :, i] = torch.randint(low=0, high=100,  size=match_shape[:, :, i].shape, device="cuda:0")   
        return return_int_str 

    def rounding_forward_hook_64(self, name, module, input, result_old):
        result = result_old
        if("transformer" in name and "drop" in name):
            int_mask = self.get_random_int_str(input[0])
            mask = (int_mask > int(100*self.DropoutP)).float() / (1 - self.DropoutP)
            result = CustomDropoutFunction.apply(input[0], mask)
        if("transformer" in name):
            torch.manual_seed(self.dropout_seed)
            self.dropout_seed += 1
        rounded_log = None
        if self.rounding and self.forward_audit:
            new_result, rounded_log, threshold_counters = round_val_64_to_32(result, self.rounder, use_log=self.forward_audit_reader, name=name)
            self.update_threshold_counters(threshold_counters)
        elif self.rounding:
            new_result, rounded_log, threshold_counters = round_val_64_to_32(result, self.rounder, name=name)
            self.update_threshold_counters(threshold_counters)
        else:
            new_result = result
        if self.do_log:
            self.forward_logger.put_tensor(rounded_log)
        return new_result.clone()

    def add_forward_hooks_64(self, model):
        hooks = []
        for name, module in model.named_modules():
            if(sum(1 for _ in module.children()) == 0):
                hooks.append(module.register_forward_hook(partial(self.rounding_forward_hook_64, "forward "+name)))
        return hooks


    def add_backward_hooks_64(self, model):
        hooks = []
        for name, module in model.named_modules():
            if(sum(1 for _ in module.children()) == 0):
                hooks.append(module.register_full_backward_hook(partial(self.full_rounding_backward_hook_64, "backward "+name)))
        return hooks
        

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()


def round_val_64_to_32(val, rounder, use_log=None, name=""):
    """ round 64 bit to 32 bit, then recast to 64 bit """
    if not val.dtype == torch.float64:
        raise ValueError("Input must be of type torch.float64.")
    
    val_flat = val.flatten()
    return_val = rounder.apply_round(val.clone()) 
    return_val_flat = return_val.flatten()
    rounded_log = (return_val_flat > val_flat).to(dtype=torch.uint8) - (return_val_flat < val_flat).to(dtype=torch.uint8) + 1 #1 if no round, 2 or 0 if round
    logging_threshold_bool = rounder.check_logging_threshold(val_flat, return_val_flat, name)
    rounded_log = torch.where(logging_threshold_bool, rounded_log, 1) # save rounded log
    counters = rounder.get_logging_condition_counts(val_flat, return_val_flat, name)
   
    if(use_log):
        val_flat_clone = val_flat.clone()
        logging_threshold_bool = rounder.check_logging_threshold(val_flat, return_val_flat, name)

        udir_0,udir_2 = use_log.verify_tensor_final(rounded_log)
        cond_down = torch.logical_and(rounded_log==2, torch.logical_and(logging_threshold_bool,udir_0))
        cond_up = torch.logical_and(rounded_log==0, torch.logical_and(logging_threshold_bool,udir_2))
        cond_down_indices = torch.nonzero(cond_down) 
        cond_up_indices = torch.nonzero(cond_up)
        if(torch.sum(cond_down) != 0 or torch.sum(cond_up) != 0):
            print("Performing Rev Correction")
            print("(Round Down, Round Up)")
            print(name)
            print((torch.sum(cond_down), torch.sum(cond_up)))
            print(cond_down_indices[:5])
            print(cond_up_indices[:5])

        original_return_val_flat = return_val_flat.clone()
        # handle round down case
        return_val_flat_down = Round64Down.apply(val_flat_clone, original_return_val_flat, rounder)
        return_val_flat = torch.where(cond_down, return_val_flat_down, return_val_flat)
        # handle round up case
        return_val_flat_up = Round64Up.apply(val_flat_clone, original_return_val_flat, rounder)
        return_val_flat = torch.where(cond_up, return_val_flat_up, return_val_flat)
        
        return_val_shape = return_val.shape
        return_val = return_val_flat.reshape(return_val_shape)
        del val_flat_clone
    del return_val_flat
    del val_flat
    return return_val, rounded_log, counters   
    

def round_loss_by_type(loss, curr_type=torch.float32): 
    if(curr_type == torch.float32):
        rounded_loss =  loss.to(dtype=torch.float16).to(dtype=torch.float32)
        return rounded_loss
    elif(curr_type == torch.float64):
        rounded_loss = loss.to(dtype=torch.float32).to(dtype=torch.float64)
        return rounded_loss
    else:
        print("No valid data type for rounding loss!")
        pass