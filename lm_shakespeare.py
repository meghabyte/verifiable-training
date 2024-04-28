import torch
import wandb
import time
from functools import partial
import os
import utils as utils
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import merkle as merkle
import json
from datetime import datetime

class ShakespeareDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_length):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        line = self.text_list[idx]
        tokens = self.tokenizer.encode(line, max_length=self.max_length, padding="max_length", truncation=True)
        tensor = torch.tensor(tokens)
        return tensor

class LMShakespeare:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.use_wandb = kwargs["use_wandb"]
        self.device = kwargs["device"]
        self.do_print = kwargs['do_print']
        self.task_name = kwargs['task']
        self.save_checkpoints = kwargs['save_checkpoints']
        self.logs_list = []
        self.loss_list = []
        if not os.path.exists("checkpoints/"+wandb.run.name):
            os.makedirs("checkpoints/"+wandb.run.name)
        self.round_loss_log = "checkpoints/"+wandb.run.name+"/loss_log"
        self.round_weights_log = "checkpoints/"+wandb.run.name+"/weights_log"
        self.round_forward_hooks_log = "checkpoints/"+wandb.run.name+"/forward_hooks_log"
        self.round_backward_hooks_log = "checkpoints/"+wandb.run.name+"/backward_hooks_log"
        self.hook_creator = None
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text = None
        self.input_ids = None
        self.train_dataset = None
        self.hash_interval = -1
        self.merkle_tree_leaves = []
        
    def set_hash_interval(self, hash_interval):
        self.hash_interval = hash_interval
        
        
    def set_hook_creator(self, forward_audit=None, back_audit=None, round_amount=32):
        self.hook_creator = utils.HookCreator(check_logs=None,
                                              rounding=self.args["rounding"],
                                              do_log=self.do_print,
                                              forward_log_fn=self.round_forward_hooks_log,
                                              backward_log_fn=self.round_backward_hooks_log,
                                              back_audit=back_audit,
                                              forward_audit=forward_audit,
                                              round_amount=round_amount)
        
    def load_data(self):
        print("Loaded shakespeare data!")
        with open('datasets/shakespeare.txt', 'r', encoding='utf-8') as f:
            self.text = f.readlines()
        self.text = [x for x in self.text if len(x.strip()) != 0]
        self.max_len =  64 # max token length
        self.train_dataset = ShakespeareDataset(self.text , self.tokenizer, max_length= self.max_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args["train_batch"], shuffle=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        return torch.stack(batch)
    
    def load_model(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if(self.args["precision"] == "16"):
            utils.convert_to_half_precision(self.model)
        elif(self.args["precision"] == "64"):
            utils.set_model_type(self.model, torch.float64)
        else: 
            utils.set_model_type(self.model, torch.float32)
        if(self.args["precision"] == "64"):
            self.model_backward_hooks = self.hook_creator.add_backward_hooks_64(self.model)
            self.model_forward_hooks = self.hook_creator.add_forward_hooks_64(self.model)
        elif(self.args["precision"] == "32"):
            self.model_backward_hooks = self.hook_creator.add_backward_hooks_32(self.model) 
            self.model_forward_hooks = self.hook_creator.add_forward_hooks_32(self.model)
        
    def train(self, epoch):
        if self.do_print:
            print("\nEpoch: %d" % epoch)
            
        self.model.train()
        train_loss, total, correct = 0, 0, 0
        for batch_idx, batch in enumerate(self.train_loader):
            if(self.args["num_batches"] != 0 and batch_idx >= self.args["num_batches"]):
                continue
            if(self.hash_interval != -1 and  batch_idx % self.hash_interval == 0):
                self.merkle_tree_leaves.append(merkle.create_str(self.model.state_dict()))
            
            self.hook_creator.hook_creator_update_fns(batch_fn=batch_idx, epoch_fn=epoch)
            if self.do_print and self.args["rounding"]:
                self.hook_creator.start_rounding_logs()
            #Audit readers
            self.hook_creator.start_audit_readers()
            
            var_backward_hooks = []
            batch_input_ids = batch[:, :-1].contiguous()
            batch_labels = batch[:, 1:].contiguous()
            if(batch_input_ids.size(1) == 0):
                continue
            inputs, targets = batch_input_ids.to(self.device), batch_labels.to(self.device)

            if 'cuda' in self.device:
                torch.cuda.synchronize()
            self.optimizer.zero_grad()
            if 'cuda' in self.device:
                torch.cuda.synchronize()

            # forward
            outputs = self.model(input_ids=inputs, labels=targets)
            if(self.args["precision"] == "64"):
                var_backward_hooks.append(outputs[1].register_hook(partial(self.hook_creator.rounding_backward_hook_64, "output")))
            elif(self.args["precision"] == "32"):
                var_backward_hooks.append(outputs[1].register_hook(partial(self.hook_creator.rounding_backward_hook_32, "output")))

            loss = outputs[0]
            print(loss.item())
            if(self.args["precision"] == "64"):
                var_backward_hooks.append(loss.register_hook(partial(self.hook_creator.rounding_backward_hook_64, "loss")))
            elif(self.args["precision"] == "32"):
                var_backward_hooks.append(loss.register_hook(partial(self.hook_creator.rounding_backward_hook_32, "loss")))
            if(self.args["rounding"] and self.args["precision"] == "64"):
                loss = utils.round_loss_by_type(loss, torch.float64)
            elif(self.args["rounding"] and self.args["precision"] == "32"):
                loss = utils.round_loss_by_type(loss, torch.float32) 

            # backprop
            print(loss.item())
            print("Back Prop")
            loss.backward()
            self.optimizer.step() 

            print(loss.item())
            train_loss += loss.item()
            self.loss_list.append(utils.float_to_binary_64(loss.detach().cpu()))

            if self.do_print and self.args["rounding"]:
                self.hook_creator.save_rounding_logs()
                print("Saving rounding logs")
                print("\n")
                
            #clear audit readers
            self.hook_creator.del_audit_readers()

            _, predicted = outputs[1].max(2)
            total += targets.size(1)*targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_accuracy = 100.*correct/total

            print((epoch, batch_idx, len(self.train_dataset), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % ((train_loss/(batch_idx+1)), train_accuracy, correct, total)))
            
            # Remove hooks
            for h in var_backward_hooks:
                h.remove()

            if self.use_wandb:
                wandb.log(
                    {"train_loss":train_loss/(batch_idx+1), 
                    "train_acc":train_accuracy}
                )
            
    def save(self):
        print("MERKLE")
        merkle_tree = merkle.create_tree(self.merkle_tree_leaves)
        if(self.args["path"]):   
            if(merkle_tree):
                print(len(self.merkle_tree_leaves))
                wandb.log({"merkle_root":merkle_tree.get_state().hex()})
                with open("checkpoints/"+wandb.run.name+"_merkle", "w") as treef:
                    leaves_list = merkle.get_list_leaves(merkle_tree)
                    json.dump(leaves_list, treef)
            fn = self.args["path"]+"_"+str(int(time.time()))
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'hooks':self.hook_creator.curr_hook_logs,
                'loss':self.loss_list,
                'fn':fn
                }, "checkpoints/"+wandb.run.name+"_ckpt")
            
            if self.save_checkpoints:
                wandb.save("checkpoints/"+wandb.run.name+"/ckpt")
                
    def run(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args["lr"])
        total_steps = len(self.train_dataset)  * self.args["epochs"] // self.args["train_batch"]
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        for epoch in range(self.args["epochs"]):
            self.train(epoch)
            self.scheduler.step()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print("Removing Model Hooks")
        self.hook_creator.remove_hooks(self.model_backward_hooks)
        self.hook_creator.remove_hooks(self.model_forward_hooks)
