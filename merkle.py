"""
Utilities for building Merkle tree for storing model checkpoints.
"""
from pymerkle import InmemoryTree as MerkleTree
import torch
import sys
import hashlib
import json


def tensor_to_numpy(dictionary):
    m = hashlib.sha256()
    return_dictionary = {}
    count = 0
    for (k,v) in sorted(dictionary.items()):
        if("masked" in k or ".attn.bias" in k): #some transformer package versions ignore this
            continue
        count += 1
        m.update(v.to(torch.float32).flatten().view(torch.int32).cpu().numpy().tobytes())
        hash_k = m.hexdigest()
        return_dictionary[k] = hash_k
    return return_dictionary
        
        
def test_hash():
    import transformers
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to("cpu")
    model.to(torch.float64)
    state_dict = tensor_to_numpy(model.state_dict())
    dict_str = str(state_dict).encode('UTF-8')
    tree = MerkleTree(algorithm='sha256')
    index = tree.append_entry(dict_str)
    value = tree.get_leaf(index)
    print(value)
    # should be b'\xa3wuOt\x8d\x87\x0eBJ\xc8sa\xc4.\x9d\xc0\x14C/I\xaa(\xe9\xde\x84\xed\xd4\xechs\xfc'


def create_str(model_state_dict):
    checkpoint_dict = tensor_to_numpy(model_state_dict)
    dict_str = str(checkpoint_dict).encode('UTF-8')
    return dict_str
    
    
def get_list_leaves(input_tree):
    leaves_list = []
    for i in range(1, input_tree.get_size()+1):
        leaves_list.append(input_tree.get_leaf(i).hex())
    return leaves_list
            
def create_tree(leaf_list):
    if(len(leaf_list) == 0):
        return None
    tree = MerkleTree(algorithm='sha256')
    for leaf in leaf_list:
        tree.append_entry(leaf)
    print("Tree Size: ")
    print(tree.get_size())
    print("Tree State: ")
    print(tree.get_state())
    print(get_list_leaves(tree))
    return tree
    
