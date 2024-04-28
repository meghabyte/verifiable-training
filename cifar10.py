import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tasks import ClassificationTask
import utils
import merkle
from datetime import datetime



class CIFAR10(ClassificationTask):
    def __init__(self, **kwargs):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.model_backward_hooks = []
        self.model_forward_hooks = []
        super().__init__(**kwargs)
        
    def load_model(self):
        self.model = resnet.ResNet50() #for VGG experiments use: torchvision.smodels.vgg11(pretrained=True); utils.disable_inplace_activation(vgg11);
        self.model.to(self.device)
        if(self.args["precision"] == "16"):
            utils.convert_to_half_precision(self.model)
        elif(self.args["precision"] == "64"):
            utils.set_model_type(self.model, torch.float64)
        else: # otherwise set to 32 bit float precision
            utils.set_model_type(self.model, torch.float32)
        if(self.args["precision"] == "64"):
            self.model_backward_hooks = self.hook_creator.add_backward_hooks_64(self.model)
            self.model_forward_hooks = self.hook_creator.add_forward_hooks_64(self.model)
        elif(self.args["precision"] == "32" and self.args["rounding"] == 1):
            self.model_backward_hooks = self.hook_creator.add_backward_hooks_32(self.model) 
            self.model_forward_hooks = self.hook_creator.add_forward_hooks_32(self.model)

        
    def load_data(self):      
        # load transforms
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # load data 
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args["train_batch"], shuffle=True, num_workers=1)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)      
            
            
    def run(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args["lr"], momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        for epoch in range(self.args["epochs"]):
            self.train(epoch)
            #self.test() enable to test model
            self.scheduler.step()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        if(self.args["rounding"] == 1):
            print("Removing Model Hooks")
            self.hook_creator.remove_hooks(self.model_backward_hooks)
            self.hook_creator.remove_hooks(self.model_forward_hooks)
            
    