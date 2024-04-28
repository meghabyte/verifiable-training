import torch
import wandb
import time
from functools import partial
torch.set_printoptions(precision=10)
import os
import utils as utils
import json
import merkle as merkle


class ClassificationTask:
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
        self.hash_interval = None
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
        
    def train(self, epoch):
        if self.do_print:
            print("\nEpoch: %d" % epoch)

        self.model.train()
        train_loss, total, correct = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if(self.args["num_batches"] != 0 and batch_idx >= self.args["num_batches"]):
                continue
            if(self.hash_interval != -1 and  batch_idx % self.hash_interval == 0):
                print("add hash!")
                self.merkle_tree_leaves.append(merkle.create_str(self.model.state_dict()))
            
            self.hook_creator.hook_creator_update_fns(batch_fn=batch_idx, epoch_fn=epoch)
            
            if self.do_print and self.args["rounding"]:
                self.hook_creator.start_rounding_logs()
            #Audit readers
            self.hook_creator.start_audit_readers()
            
            var_backward_hooks = []

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.task_name not in ['lm_synthetic']:
                if(self.args["precision"] == "16"):
                    inputs = inputs.to(torch.float16)
                elif(self.args["precision"] == "64"):
                    inputs = inputs.to(torch.float64)
                else:
                    inputs = inputs.to(torch.float32)

            if 'cuda' in self.device:
                torch.cuda.synchronize()
            self.optimizer.zero_grad()
            if 'cuda' in self.device:
                torch.cuda.synchronize()

            # forward
            outputs = self.model(inputs)
            if(self.args["precision"] == "64" and self.args["rounding"] == 1):
                var_backward_hooks.append(outputs.register_hook(partial(self.hook_creator.rounding_backward_hook_64, "output")))
            elif(self.args["precision"] == "32" and self.args["rounding"] == 1):
                var_backward_hooks.append(outputs.register_hook(partial(self.hook_creator.rounding_backward_hook_32, "output")))

            # loss computation
            if self.task_name in ['lm_synthetic']:
                outputs = outputs[0]
                outputs = rearrange(outputs, '... C -> (...) C')
                targets = rearrange(targets, '... -> (...)')
                
            #print("Loss")

            loss = self.loss_fn(outputs, targets)
            if(self.args["precision"] == "64" and self.args["rounding"] == 1):
                var_backward_hooks.append(loss.register_hook(partial(self.hook_creator.rounding_backward_hook_64, "loss")))
            elif(self.args["precision"] == "32" and self.args["rounding"] == 1):
                var_backward_hooks.append(loss.register_hook(partial(self.hook_creator.rounding_backward_hook_32, "loss")))
            if(self.args["rounding"] and self.args["precision"] == "64"):
                loss = utils.round_loss_by_type(loss, torch.float64)
            elif(self.args["rounding"] and self.args["precision"] == "32"):
                loss = utils.round_loss_by_type(loss, torch.float32) 
            if 'cuda' in self.device:
                torch.cuda.synchronize()
            loss.backward()
            self.optimizer.step() 
            if 'cuda' in self.device:
                torch.cuda.synchronize()

            
            train_loss += loss.item()

            if self.do_print and self.args["rounding"]:
                self.hook_creator.save_rounding_logs()
                
            #clear audit readers
            self.hook_creator.del_audit_readers()
            
            if 'cuda' in self.device:
                torch.cuda.synchronize()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_accuracy = 100.*correct/total

            print((epoch, batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % ((train_loss/(batch_idx+1)), train_accuracy, correct, total)))
            
            # Remove hooks
            for h in var_backward_hooks:
                h.remove()

        print("add hash!")
        self.merkle_tree_leaves.append(merkle.create_str(self.model.state_dict()))
                
        if self.use_wandb:
            wandb.log(
                {"train_loss":train_loss/batch_idx+1, 
                "train_acc":train_accuracy}
            )
        
    def test(self):
        self.model.eval()
        test_loss, total, correct = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if(self.args["precision"] == "16"):
                    inputs = inputs.to(torch.float16)
                elif(self.args["precision"] == "64"):
                    inputs = inputs.to(torch.float64)
                elif(self.args["precision"] == "half"):
                    inputs = inputs.half()
                else:
                    print(f"32 precision")


                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                
                loss = self.loss_fn(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                test_accuracy = 100.*correct/total
                print((batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % ((test_loss/(batch_idx+1)), test_accuracy, correct, total)))
            
            if self.use_wandb:
                wandb.log(
                    {
                        "test_loss":test_loss/batch_idx+1, 
                        "test_acc":test_accuracy
                    }
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
