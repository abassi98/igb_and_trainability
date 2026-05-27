"""PyTorch and Lightning model definitions used across initialization and training.

The module defines a shared Lightning base class that handles logging and hook
registration, plus concrete architectures ranging from simple MLPs and CNNs to
transformers and pretrained timm backbones.
"""


import torch
import torch.nn as nn
import lightning as pl
import numpy as np
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassAccuracy
from src.train_utils import MultiClassFreq, MultiClassLoss, ElementsPerClass, delta_orthogonal_
from src.utils import calculate_gain
import math 
from timm.models import create_model

class SuperModule(pl.LightningModule):
    """Common Lightning base class for all trainable models in the repository.

    It centralizes optimizer setup, per-class metrics, and forward/backward hook
    registration used to measure initialization-time gradient statistics.
    """
    def __init__(self,
                sigma_w,
                sigma_b,
                lr = 0.001, 
                n_classes=10, 
                weight_decay=0.0):
        super(SuperModule, self).__init__()
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.lr = lr
        self.n_classes = n_classes
        self.weight_decay = weight_decay
  
        # save accuracy per class, frequency per class and loss per class during training
        # train
        self.train_elements_per_class = ElementsPerClass(self.n_classes)
        self.train_acc_per_class = MulticlassAccuracy(num_classes=self.n_classes, average=None)
        self.train_freq_per_class = MultiClassFreq(self.n_classes, device=self.device)
        self.train_loss_per_class = MultiClassLoss(self.n_classes, device=self.device) 
        
        # test
        self.test_elements_per_class = ElementsPerClass(self.n_classes)
        self.test_acc_per_class = MulticlassAccuracy(num_classes=self.n_classes, average=None)
        self.test_freq_per_class = MultiClassFreq(self.n_classes, device=self.device)  # use self.device to ensure it works on GPU if needed
        self.test_loss_per_class = MultiClassLoss(self.n_classes, device=self.device) 

        # loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

        # save stats of gradients per class
        self.registered_names = []
        self.registered_layers = 0
        self.batch_grad_storage = {}
        self.batch_preact_storage = {}
        self.current_batch_y = None
        self.batch_size = None
        self.non_linearities = []
        
    def _init(self):
        """Dispatch initialization according to the selected initialization mode."""
        # initialize
        if self.init is not None:
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()
        else:
            # init only the head
            std = self.sigma_w / np.sqrt(self.model.head.in_features)
            nn.init.normal_(self.model.head.weight, mean=0.0, std=std)
            nn.init.normal_(self.model.head.bias, mean=0.0, std=self.sigma_b)

    def _init_gaussian(self):
        """
        Gaussian initialization
        """ 
        i = 0 # counter for registered modules
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                gain = np.sqrt(calculate_gain(self.non_linearities[i], Vb=self.sigma_b**2))
                std = gain * self.sigma_w / np.sqrt(module.in_features)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=self.sigma_b)
                i += 1 # update coounter
            elif isinstance(module, nn.Conv2d):
                gain = np.sqrt(calculate_gain(self.non_linearities[i], Vb=self.sigma_b**2))
                std = gain * self.sigma_w * np.sqrt(module.stride[0] * module.stride[1]) / np.sqrt(module.kernel_size[0] * module.kernel_size[1] * module.out_channels)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=self.sigma_b)
                i += 1 # update coounter
            
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            

    def _init_orthogonal(self):
        """
        Orthogonal initialization (dynamical isometry)
        """
        i = 0 # counter for registered modules
        for _, module in enumerate(self.named_modules()):
            if isinstance(module, nn.Linear):
                gain = np.sqrt(calculate_gain(self.non_linearities[i]))
                std = gain * self.sigma_w / np.sqrt(module.in_features)
                nn.init.orthogonal_(module.weight, gain=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                i += 1 # update coounter
            elif isinstance(module, nn.Conv2d):
                gain = np.sqrt(calculate_gain(self.non_linearities[i]))
                delta_orthogonal_(module.weight, gain=self.sigma_w)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                i += 1 # update coounter
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
            
    # save pre-activations interesting quantities
    def save_pre_activations(self, layer_name):
        """Create a forward hook that stores flattened pre-activations by layer."""
        def hook(module, input, output):
            # 'output' here is the pre-activation if activation is applied separately
            # Detach to avoid storing computational graph
            preact = output.detach()
            # Flatten spatial dimensions if needed (e.g., for Conv2d)
            pre_act_flat = torch.flatten(preact, start_dim=1)
            self.batch_preact_storage[layer_name] = pre_act_flat.detach()
        return hook

    def save_forward_hook(self):
        """Register forward hooks on linear and convolutional layers."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self.save_pre_activations(name))

    # save gradients norm
    def save_pre_activation_grads(self, layer_name):
        """Create a backward hook that stores samplewise squared gradient norms."""
        def hook(module, grad_input, grad_output):
            # grad_output is a tuple: gradients w.r.t. the output of this module
            grad = grad_output[0]  # shape [batch, features]
            # mean squared grad per sample
            self.batch_grad_storage[layer_name] = torch.mean(torch.flatten(grad.detach(), start_dim=1)**2, dim=1)
        return hook
    
    def save_backward_hook(self):
        """Register backward hooks on all linear and convolutional layers."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.registered_names.append(name)
                module.register_full_backward_hook(self.save_pre_activation_grads(name))

        self.registered_layers = len(self.registered_names)
        
    def training_step(self, batch, batch_idx):
        """Run one training step and update classwise training metrics."""
        x, y = batch
        #print("x, y shape", x.shape, y.shape)
        self.batch_size = x.shape[0]
        self.current_batch_y = y
        y_hat = self(x)
        #print("y_hat", y_hat.shape, y_hat)
        losses = self.loss_fn(y_hat, y)  # Classification-friendly
        loss = torch.mean(losses)

        self.train_acc_per_class.update(y_hat, y)  # compute accuracy per-class
        self.train_freq_per_class.update(y_hat)  # update frequency per-class
        self.train_loss_per_class.update(losses, y)  # update losses per-class
        self.train_elements_per_class.update(y)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        """Run one validation step and update classwise validation metrics."""
        x, y = batch
        y_hat = self(x)
        losses = self.loss_fn(y_hat, y)  # Classification-friendly
        loss = torch.mean(losses)

        self.test_acc_per_class.update(y_hat, y)  # compute accuracy per-class
        self.test_freq_per_class.update(y_hat)  # update frequency per-class
        self.test_loss_per_class.update(losses, y)  # update losses per-class
        self.test_elements_per_class.update(y)

        return loss
   
    def on_validation_epoch_end(self):
        """Aggregate, log, and reset train/validation metrics at epoch end."""
        # train
        train_accs = self.train_acc_per_class.compute()
        train_freqs = self.train_freq_per_class.compute()
        train_losses = self.train_loss_per_class.compute()
        train_elements_per_class = self.train_elements_per_class.compute()
        ### accumulate metrics
        self.train_freqs = train_freqs
        train_acc = torch.sum(train_accs.cpu() * train_elements_per_class)/torch.sum(train_elements_per_class)
        train_loss = torch.sum(train_losses * train_elements_per_class)/torch.sum(train_elements_per_class)
        # test
        test_accs = self.test_acc_per_class.compute()
        test_freqs = self.test_freq_per_class.compute()
        test_losses = self.test_loss_per_class.compute()
        test_elements_per_class = self.test_elements_per_class.compute()
        test_acc = torch.sum(test_accs.cpu() * test_elements_per_class)/torch.sum(test_elements_per_class)
        test_loss = torch.sum(test_losses * test_elements_per_class)/torch.sum(test_elements_per_class)

        # log overall metrics
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True) # overall accuracy
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", test_acc, on_step=False, on_epoch=True, prog_bar=True) # overall accuracy
        
        # log per class metrics
        for c in range(self.n_classes):
            self.log(f"train_loss_class_{c}", train_losses[c], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train_acc_class_{c}", train_accs[c], on_step=False, on_epoch=True, prog_bar=True) # accuracy per class
            self.log(f"train_freq_class_{c}", train_freqs[c], on_step=False, on_epoch=True, prog_bar=True) # frequency per class
            self.log(f"test_loss_class_{c}", test_losses[c], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"test_acc_class_{c}", test_accs[c], on_step=False, on_epoch=True, prog_bar=True) # accuracy per class
            self.log(f"test_freq_class_{c}", test_freqs[c], on_step=False, on_epoch=True, prog_bar=True) # frequency per class
            

        # reset
        self.train_acc_per_class.reset()
        self.train_freq_per_class.reset()
        self.train_loss_per_class.reset()
        self.train_elements_per_class.reset()
        self.test_acc_per_class.reset()
        self.test_freq_per_class.reset()
        self.test_loss_per_class.reset()
        self.test_elements_per_class.reset()

        # save gradient per class, averaged over the depth to have an aggregated metric
        if self.registered_layers > 0:
            if self.current_batch_y is not None:
                grads_all = np.zeros((self.batch_size, len((self.batch_grad_storage.items()))))                   
                for nl, (layer_name, grads) in enumerate(self.batch_grad_storage.items()):
                        #print(nl)
                        grads_all[:, nl] = grads.cpu().numpy() * self.batch_size**2 # rescale for batch size since reduction is mean
                
                y = self.current_batch_y
                # Group gradients by class
                for c in range(self.n_classes):
                    mask = (y == c)
                    if mask.any():
                        grads_class = grads_all[mask.cpu().numpy(),:]
                        #print(np.max(grads_all, axis=1)/np.min(grads_all, axis=1))
                        #print(np.mean(np.log10(grads_all), axis=1), np.var(np.log10(grads_all), axis=1))
                        #print(f"init grads - class f{c}", np.mean(grads_all[:,-1]), np.std(grads_all[:,-1]))
                        #print(np.log10(grads_class))
                        mean_grads = grads_class[:,-1]   # take gradient of first layer
                        #print(mean_grads)
                        self.log(f"grads_mean_class_{c}", np.mean(mean_grads),  on_step=False, on_epoch=True, prog_bar=False)# 5 % quantile
                        self.log(f"grads_5q_class_{c}", np.quantile(mean_grads, 0.05),  on_step=False, on_epoch=True, prog_bar=False)# 5 % quantile
                        self.log(f"grads_25q_class_{c}", np.quantile(mean_grads, 0.25),  on_step=False, on_epoch=True, prog_bar=False)# 25 % quantile
                        self.log(f"grads_median_class_{c}", np.quantile(mean_grads, 0.5),  on_step=False, on_epoch=True, prog_bar=False)# median
                        self.log(f"grads_75q_class_{c}", np.quantile(mean_grads, 0.75),  on_step=False, on_epoch=True, prog_bar=False)# 75 % quantile
                        self.log(f"grads_95q_class_{c}", np.quantile(mean_grads, 0.95),  on_step=False, on_epoch=True, prog_bar=False)# 95 % quantile
               
    def configure_optimizers(self):
        """Use Adam with the learning rate and weight decay stored on the module."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
### Standard MLP perceptron with fixed width
class MLP(SuperModule):
    """Fully connected network used for both plain and residual MLP experiments."""
    def __init__(self, 
                input_size,
                hidd_layer_size,
                hidd_layers,
                act,
                act2,
                sigma_w,
                sigma_b,
                lr = 0.001, 
                n_classes=10,
                residual=False,
                init="Gaussian"):
        
        super(MLP, self).__init__(sigma_w, sigma_b, lr, n_classes)
        self.input_size = input_size
        self.hidd_layer_size = hidd_layer_size
        self.hidd_layers = hidd_layers
        self.act = act
        self.act2 = act2
        self.residual = residual
        self.init = init
        
        # Hidden layers
        layers=[]
        layers.append(nn.Linear(self.input_size, self.hidd_layer_size))
        for _ in range(1,self.hidd_layers):
            layers.append(nn.Linear(self.hidd_layer_size, self.hidd_layer_size))
        
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(self.hidd_layer_size,self.n_classes)

        # save forward/backward hooks
        self.save_forward_hook()
        self.save_backward_hook()

        # non linearities
        self.non_linearities = []
        for _ in range(self.hidd_layers+1):
            self.non_linearities.append(self.act.__class__.__name__)

        # initialize
        if self.init is not None:
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()
    

    def forward(self, x):
        """Propagate inputs through the hidden stack and classifier head."""
        x = self.layers[0](x)
        for l in self.layers[1:]:
            l_x = l(x)
            if self.residual:
                x = x + l_x / math.sqrt(self.hidd_layers)
            else:
                x = l_x
            
            x = self.act(x)
            if self.act2 is not None:
                x = self.act2(x)
                stacked = torch.stack((x, x), dim=2)
                x = stacked.view(x.shape[0], -1) 

        return self.out(x)
    


### Convolutional CNN (Xiao et al., 2019s)
class CNN(SuperModule):
    """Deep convolutional network used in initialization and training studies."""
    def __init__(self,
                fig_size=28,
                in_channels=3,
                hidd_channels=300,
                hidd_layers=100,
                act=nn.ReLU(),
                sigma_w=1.0,
                sigma_b=0.0,
                lr = 0.001, 
                n_classes=10, 
                init="Gaussian"):
        super(CNN, self).__init__(sigma_w, sigma_b, lr, n_classes)

        self.fig_size = fig_size
        self.in_channels = in_channels
        self.hidd_channels = hidd_channels
        self.hidd_layers = hidd_layers
        self.act = act
        self.act = act
        self.init = init

        # Input layers
        layers=[]
        layers.append(nn.Conv2d(self.in_channels, self.hidd_channels, kernel_size=3, padding=0, stride=1))
        out_size = math.floor(self.fig_size - 2)
        layers.append(nn.Conv2d(self.hidd_channels, self.hidd_channels, kernel_size=3, padding=0, stride=2))
        out_size = math.floor((out_size - 3)/2+1)
        layers.append(nn.Conv2d(self.hidd_channels, self.hidd_channels, kernel_size=3, padding=0, stride=2))
        out_size = math.floor((out_size - 3)/2+1)

        # Hidden layers
        for _ in range(0,self.hidd_layers-3):
            layers.append(nn.Conv2d(self.hidd_channels, self.hidd_channels, kernel_size=3, padding=1, stride=1, padding_mode='circular'))
        
        #self.avg_pool = nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.out = nn.Linear(self.hidd_channels*out_size*out_size, self.n_classes)
        self.layers = nn.Sequential(*layers)
       
        # save forward/backward hooks
        self.save_forward_hook()
        self.save_backward_hook()

        # non linearities
        self.non_linearities = ["Linear", "Linear"]
        for _ in range(self.hidd_layers):
            self.non_linearities.append(self.act.__class__.__name__)

        # initialize
        if self.init is not None:
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()
        
    def forward(self, x):
        """Apply convolutional blocks followed by flattening and classification."""
        for l in self.layers:
            x = l(x)
            x = self.act(x)

        #x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.out(x)
        
        return x
    
# pretrained large VIT
class LargeVIT(SuperModule):
    """Wrapper around a timm ViT-L/16 backbone with optional custom initialization."""
    def __init__(self, 
                sigma_w=1.0,
                sigma_b=0.0,
                lr = 0.001,
                n_classes = 200,
                save_hooks=False,
                weight_decay=0.0,
                init=None,
                train=False):
        super(LargeVIT, self).__init__(sigma_w, sigma_b, lr, n_classes, weight_decay)
        self.init = init

        if self.init == None:
            self.model = create_model('vit_large_patch16_384', pretrained=True, drop_path_rate=0.1)
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    module.weight.data = module.weight.data * self.sigma_w
                    #pass
            if train:
                for param in self.model.parameters(): 
                 # froze model for fine tuning if train mode
                    param.requires_grad = False
                #print(param)
                #param = param * sigma_w      
            #in_features = self.model.get_classifier().in_features
            #self.model.head = nn.Linear(in_features, self.n_classes)
            self.model.reset_classifier(num_classes=n_classes)
        else:
            self.model = create_model('vit_large_patch16_384', pretrained=False, drop_path_rate=0.1)
            self.non_linearities = ["Linear"] *10000 # to be modified
            self.model.reset_classifier(num_classes=n_classes)
            # initialize
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()

        
        if save_hooks: # do not save during training
            # save forward/backward hooks
            #self.save_forward_hook()
            self.save_backward_hook()

        
        
    def forward(self, x):
        """Delegate the forward pass to the wrapped timm model."""
        x = self.model(x)
        return x
    



# pretrained large VIT
class RESNET(SuperModule):
    """Wrapper around a timm ResNet-18 backbone."""
    def __init__(self, 
                sigma_w=1.0,
                sigma_b=0.0,
                lr = 0.001,
                n_classes = 200,
                save_hooks=False,
                weight_decay=0.0,
                init=None,
                train=False):
        super(RESNET, self).__init__(sigma_w, sigma_b, lr, n_classes, weight_decay)
        self.init = init

        if self.init == None:
            self.model = create_model('resnet18', pretrained=True, drop_path_rate=0.1)
            # for name, module in self.named_modules():
            #     if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            #         module.weight.data = module.weight.data * self.sigma_w
            #         #pass
            if train:
                for param in self.model.parameters(): 
                 # froze model for fine tuning if train mode
                    param.requires_grad = False
                #print(param)
                #param = param * sigma_w      
        else:
            self.model = create_model('resnet18', pretrained=False, drop_path_rate=0.1)
            self.non_linearities = ["Linear"] *10000 # to be modified
            # initialize
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()

       
        if save_hooks: # do not save during training
            # save forward/backward hooks
            self.save_forward_hook()
            self.save_backward_hook()

        in_features = self.model.get_classifier().in_features
        self.model.head = nn.Linear(in_features, self.n_classes)
        
    def forward(self, x):
        """Delegate the forward pass to the wrapped timm model."""
        x = self.model(x)
        return x
    


# pretrained large VIT
class MLPMIXER(SuperModule):
    """Wrapper around a timm MLP-Mixer backbone."""
    def __init__(self, 
                sigma_w=1.0,
                sigma_b=0.0,
                lr = 0.001,
                n_classes = 200,
                save_hooks=False,
                weight_decay=0.0,
                init=None,
                train=False):
        super(MLPMIXER, self).__init__(sigma_w, sigma_b, lr, n_classes, weight_decay)
        self.init = init

        if self.init == None:
            self.model = create_model('mixer_l16_224.goog_in21k', pretrained=True, drop_path_rate=0.0)
            # for name, module in self.named_modules():
            #     if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            #         module.weight.data = module.weight.data * self.sigma_w
            #         #pass
            if train:
                for param in self.model.parameters(): 
                    param.requires_grad = False # froze model for fine tuning if train mode
                #param = param * sigma_w      
        else:
            self.model = create_model('mixer_l16_224.goog_in21k', pretrained=False, drop_path_rate=0.0)
            self.non_linearities = ["Linear"] *10000 # to be modified
            # initialize
            if self.init=="Gaussian":
                self._init_gaussian()
            elif self.init=="Orthogonal":
                self._init_orthogonal()

       
        if save_hooks: # do not save during training
            # save forward/backward hooks
            self.save_forward_hook()
            self.save_backward_hook()

        in_features = self.model.get_classifier().in_features
        self.model.head = nn.Linear(in_features, self.n_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x
