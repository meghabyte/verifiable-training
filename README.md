# Optimistic Verifiable Training by Controlling Hardware Nondeterminism


This repository contains the code for the paper [Optimistic Verifiable Training by Controlling Hardware Nondeterminism](https://arxiv.org/pdf/2403.09603.pdf) by [Megha Srivastava](https://cs.stanford.edu/~megha), [Simran Arora](https://arorasimran.com/), and [Dan Boneh](https://crypto.stanford.edu/~dabo/). In this work, we show how to eliminate hardware nondeterminism (i.e. achieve identical weights after training on two different GPU types) in order to design a verification scheme for 3rd party auditing of model training services. For any questions, please contact megha@cs.stanford.edu! 

While our code is implemented with version ``pytorch=1.13.1``, we have verified that it is compatible with later pytorch versions, such as ``pytorch=2.3.1``.

If you find this repository useful, please cite:

```
@inproceedings{srivastava2024verifiable,
    title = "Optimistic Verifiable Training by Controlling Hardware Nondeterminism",
    author = "Srivastava, Megha and Arora, Simran and Boneh, Dan",
    booktitle = "arxiv",
    year = "2024",
}
```

## Overview

Our work is built upon the ``pytorch`` framework. The results in our paper were generated using ``python=3.8.16``, ``pytorch=1.13.1, cuda=11.7.1, cudnn8.5.0``, and ``pymerkle=6.1.0``. Additionally, we use the HuggingFace ``transformers=4.36.2`` library for our experiments with GPT-2. Finally, we log and track experiments using Weights & Biases (``wandb==0.13.10``). 

The script ``experiments.py`` is the main entry point for our codebase, for both training mode and auditing mode. 

An example command for training ResNet-50 on 3 batches of the CIFAR-10 dataset is: 
``python experiments.py --task cifar10 --epochs 1 --learning_rate 0.01 --train_batch 128 --rounding 1 --precision 64 --use_wandb 1 --num_batches 3 --do_print 1 --round_amount 32`` 

To work with other classification datasets, you can extend ``tasks.py``. 

An example command for fine-tuning GPT-2 on 5 batches of the Shakespeare dataset is:
``python experiments.py --task lm_shakespeare --epochs 1 --learning_rate 0.00001 --train_batch 8 --rounding 1 --precision 64 --use_wandb 1 --num_batches 5 --do_print 1 --round_amount 32``

The flag ``do_print`` is necessary for generating logs to use in future auditing. For auditing, you will need rounding logs corresponding to both the forward and backward passes during training. These are saved in the ``checkpoints/$ID`` folder, where  ``$ID`` refers to the run name assigned by Weights & Biases. For example, if logs from training on one GPU were saved with the run name ``effortless-firefly-1838``, then auditing using these logs would be enabled by modifying the above two commands as shown below:

``python experiments.py --task cifar10 --epochs 1 --learning_rate 0.01 --train_batch 128 --rounding 1 --precision 64 --use_wandb 1 --num_batches 3 --do_print 0 --round_amount 32 --hash_interval 1 --forward_audit checkpoints/effortless-firefly-1838/forward_hooks_log --back_audit checkpoints/effortless-firefly-1838/backward_hooks_log``

``python experiments.py --task lm_shakespeare --epochs 1 --learning_rate 0.00001 --train_batch 8 --rounding 1 --precision 64 --use_wandb 1 --num_batches 5 --do_print 0 --round_amount 32 --hash_interval 1 --forward_audit checkpoints/effortless-firefly-1838/forward_hooks_log --back_audit checkpoints/effortless-firefly-1838/backward_hooks_log``



## Adding Model Checkpoints into Merkle Tree

The script ``merkle.py`` is used to help add model checkpoints to a Merkle tree. Notably, it contains the ``create_str`` function that serializes model checkpoints so that their hashes can be included as leaves of a Merkle tree. You can use ``test_hash`` to verify if the code is running properly on your end. By default, we append a model checkpoint after every training epoch.
