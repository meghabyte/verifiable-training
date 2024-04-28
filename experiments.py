import wandb
import argparse
import torch
import numpy as np
import random 
import cifar10
import lm_shakespeare
import os

os.environ["WANDB__SERVICE_WAIT"] = "300"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description = "arguments for verifiable training")
    parser.add_argument("-t", "--task", required=True, help = "task name")
    parser.add_argument("-rst", "--torch_random_seed", default=0, type=int, help = "torch random seed")
    parser.add_argument("-rsc", "--cuda_random_seed", default=0, type=int, help = "cuda random seed")
    parser.add_argument("-rsn", "--np_random_seed", default=0, type=int, help = "numpy random seed")
    parser.add_argument("-rs", "--random_seed", default=0, type=int, help = "random seed")
    parser.add_argument("-e", "--epochs", default=10, type=int, help = "number of epochs")
    parser.add_argument("-nb", "--num_batches", default=0, type=int, help = "number of batches")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float, help = "learning rate")
    parser.add_argument("-p", "--precision", default="32", help = "training precision")
    parser.add_argument("-sp", "--save_path", required=False, help = "save path for checkpoint")
    parser.add_argument("-b", "--train_batch", default=128, type=int, help = "training batch size")
    parser.add_argument("-dp", "--do_print", default=0, type=int, help = "print values during training")
    parser.add_argument("-wdb", "--use_wandb", default=1, type=int, help = "use wandb")
    parser.add_argument("-sc", "--save_checkpoints", default=1, type=int, help = "save checkpoints")
    parser.add_argument("-rdg", "--rounding", default=0, type=int, help = "rounding")
    parser.add_argument("-tf", "--tensor_float", required=False, help = "user tensor float")
    parser.add_argument("-fl", "--forward_audit", required=False, help = "forward audit log")
    parser.add_argument("-bl", "--back_audit", required=False, help = "backward audit log")
    parser.add_argument("-rdamt", "--round_amount", default=32, type=int, required=False, help = "rounding precision")
    parser.add_argument("-hi", "--hash_interval", default=-1, type=int, required=False, help = "hashing interval for hasing to merkle tree")

    args = parser.parse_args()
    if args.save_path:
        setattr(args, 'name', args.save_path)
    elif args.sp:
        setattr(args, 'name', args.sp)

    if args.use_wandb:
        wandb.init(project="verifiable", config=args)
        wandb.config.update(args)
    return args

def set_determinism(args):
    random.seed(args.random_seed)
    np.random.seed(args.np_random_seed)
    torch.manual_seed(args.torch_random_seed)
    torch.cuda.manual_seed(args.cuda_random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    torch.use_deterministic_algorithms(True) 
    if(args.tensor_float):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True 
    else:
        #disable tensor float32
        torch.backends.cuda.matmul.allow_tf32 = False 
        torch.backends.cudnn.allow_tf32 = False 

def create_task_instance(args):
    task = None
    task_name = args.task
    if(task_name=="cifar10"):
        task = cifar10.CIFAR10(device=device, task=args.task, epochs=args.epochs, lr=args.learning_rate, precision=args.precision, path=args.save_path, rounding=args.rounding, train_batch=args.train_batch, do_print=args.do_print, use_wandb=args.use_wandb, save_checkpoints=args.save_checkpoints, num_batches=args.num_batches)
    elif(task_name == "lm_shakespeare"):
        task = lm_shakespeare.LMShakespeare(device=device, task=args.task, epochs=args.epochs, lr=args.learning_rate, precision=args.precision, path=args.save_path, rounding=args.rounding, train_batch=args.train_batch, do_print=args.do_print, use_wandb=args.use_wandb, save_checkpoints=args.save_checkpoints, num_batches=args.num_batches)
    return task

if __name__ == "__main__":
    args = get_args()
    set_determinism(args)
    task = create_task_instance(args)
    if(args.forward_audit and args.back_audit):
        print("Running AUDIT Mode")
        task.set_hash_interval(args.hash_interval)
        task.set_hook_creator(forward_audit=args.forward_audit, back_audit=args.back_audit, round_amount=args.round_amount) #TODO also audit loss
    else:
        print("Running TRAIN Mode")
        print(args.round_amount)
        task.set_hash_interval(args.hash_interval)
        task.set_hook_creator(round_amount=args.round_amount)
    task.load_model()
    task.load_data()
    task.run()
    task.save()