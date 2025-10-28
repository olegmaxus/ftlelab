from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .utils import device_string
from tqdm import tqdm
import copy

LOSS_MAP = {
    "mse": nn.MSELoss,
    "bce": nn.BCEWithLogitsLoss,
    "ce": nn.CrossEntropyLoss,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW
}

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    weight_decay: float = 0.0
    loss: str = "mse"   # "mse" or "bce" for +/-1 targets, or "ce" for multi-class
    optimizer: str = "adam"
    momentum: float = 0.0
    save_dir: str = 'checkpoints'
    model_name: str = '0'
    print_every: int = 10
    # -- freezing options
    train_only_output: bool = False          # train only the final nn.Linear
    train_last_n_linears: int = 0            # train last k linear layers (e.g., k=2 = last hidden + output)
    train_param_names: tuple = ()            # explicit whitelist of parameter names to train
    freeze_param_names: tuple = ()           # explicit blacklist
    freeze_regex: str = ""  

# class Trainer:
#     def __init__(self, model: nn.Module, cfg: TrainConfig):
#         self.model = model
#         self.cfg = cfg
#         self.device = device_string()
#         self.model.to(self.device) # maybe set if requested?
#         print("Trainer for " + "MODEL_" + self.cfg.model_name +  f" was initialized on device: {self.device}")
        
#         loss_name = self.cfg.loss.lower()
#         loss_class = LOSS_MAP.get(loss_name)

#         if loss_class:
#             self.criterion = loss_class()
#         else:
#             raise ValueError(f"Unknown loss: '{self.cfg.loss}'. Supported losses are: {list(LOSS_MAP.keys())}")

#         optimizer_name = self.cfg.optimizer.lower()
#         optimizer_class = OPTIMIZER_MAP.get(optimizer_name)

#         if optimizer_class:
#             if optimizer_name == "sgd":
#                 self.optimizer = optimizer_class(self.model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
#             else:
#                 self.optimizer = optimizer_class(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
#         else:
#             raise ValueError(f"Unknown optimizer: '{self.cfg.optimizer}'. Supported: {list(OPTIMIZER_MAP.keys())}")
        
#         self.current_epoch = 0
#         self.best_val_loss = float('inf')
#         self.history = {'train_loss': [], 
#                         'val_loss': [], 
#                         'val_accuracy': []}

#         # Create save directory
#         os.makedirs(self.cfg.save_dir, exist_ok=True)

#     def _train_one_epoch(self, train_loader: DataLoader):
#         """Performs a single training epoch."""
#         self.model.train()
#         running_loss = 0.0
        
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Training]") if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs else train_loader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)
            
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
            
#         return running_loss / len(train_loader.dataset)

#     def _validate_one_epoch(self, val_loader: DataLoader):
#         """Performs a single validation epoch."""
#         self.model.eval()
#         running_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0
        
#         with torch.no_grad():
#             for inputs, labels in tqdm(val_loader, desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Validation]") if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs else val_loader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 running_loss += loss.item() * inputs.size(0)
                
#                 predicted = torch.sign(outputs) if outputs.shape[-1] == 1 else torch.argmax(outputs, dim=-1)
                
#                 total_predictions += labels.size(0)
#                 correct_predictions += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / len(val_loader.dataset)
#         epoch_acc = (correct_predictions / total_predictions) * 100
#         return epoch_loss, epoch_acc

#     def save_checkpoint(self, is_best=False):
#         state = {'epoch': self.current_epoch, 
#                  'model_state_dict': self.model.state_dict(), 
#                  'optimizer_state_dict': self.optimizer.state_dict(), 
#                  'best_val_loss': self.best_val_loss}
        
#         filepath = os.path.join(self.cfg.save_dir, 'BEST_MODEL' + self.cfg.model_name + '.pt' if is_best 
#                                 else 'LAST_CHECKPOINT'+ self.cfg.model_name + '.pt')
#         torch.save(state, filepath)

#     def train(self, train_loader: DataLoader, val_loader: DataLoader):
#         """The main training loop."""
#         print("Training started...")
#         for epoch in range(self.cfg.epochs):
#             self.current_epoch = epoch
            
#             train_loss = self._train_one_epoch(train_loader)
#             val_loss, val_acc = self._validate_one_epoch(val_loader)
            
#             self.history['train_loss'].append(train_loss)
#             self.history['val_loss'].append(val_loss)
#             self.history['val_accuracy'].append(val_acc)
            
#             if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
#                 print(f"Epoch {epoch+1}/{self.cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

#             self.save_checkpoint(is_best=False)

#             if val_loss < self.best_val_loss:
#                 if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
#                     print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving the best model.")
#                 # if (self.current_epoch + 1) % self.cfg.print_every != 0 and (self.current_epoch + 1) != self.cfg.epochs:
#                 #     print(f"Epoch {epoch+1:03d}/{self.cfg.epochs} | Found the new best model. Val Loss: {val_loss:.4f}")
#                 # else: # Otherwise, just note the improvement on the regular log line
#                 #     print(f"✨ Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")

#                 self.best_val_loss = val_loss
#                 self.save_checkpoint(is_best=True)

#         print("Training finished.")
#         return self.history


from dataclasses import dataclass
import os, re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .utils import device_string
from tqdm import tqdm

LOSS_MAP = {
    "mse": nn.MSELoss,
    "bce": nn.BCEWithLogitsLoss,
    "ce": nn.CrossEntropyLoss,
}

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW
}

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    weight_decay: float = 0.0
    loss: str = "mse"          # "mse" or "bce" for +/-1 targets, or "ce" for multi-class
    optimizer: str = "adam"    # <-- removed stray comma
    momentum: float = 0.0
    save_dir: str = "checkpoints"
    model_name: str = "0"
    print_every: int = 10

    # ---- Freezing options ----
    train_only_output: bool = False          # train only the final nn.Linear
    train_last_n_linears: int = 0            # train last k linear layers (e.g., k=2 = last hidden + output)
    train_param_names: tuple = ()            # explicit whitelist of parameter names to train
    freeze_param_names: tuple = ()           # explicit blacklist
    freeze_regex: str = ""                   # freeze any param whose name matches this regex

class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.device = device_string()
        self.model.to(self.device)

        print("Trainer for " + "MODEL_" + self.cfg.model_name +  f" was initialized on device: {self.device}")

        # --- Loss ---
        loss_name = self.cfg.loss.lower()
        loss_class = LOSS_MAP.get(loss_name)
        if not loss_class:
            raise ValueError(f"Unknown loss: '{self.cfg.loss}'. Supported: {list(LOSS_MAP.keys())}")
        self.criterion = loss_class()

        # --- Apply freezing policy BEFORE building optimizer ---
        self._apply_freezing_policy()

        # --- Optimizer over trainable params only ---
        optimizer_name = self.cfg.optimizer.lower()
        opt_cls = OPTIMIZER_MAP.get(optimizer_name)
        if not opt_cls:
            raise ValueError(f"Unknown optimizer: '{self.cfg.optimizer}'. Supported: {list(OPTIMIZER_MAP.keys())}")

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters after applying the freezing policy.")

        if optimizer_name == "sgd":
            self.optimizer = opt_cls(params, lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        else:
            self.optimizer = opt_cls(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # --- Bookkeeping ---
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        # Small report
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    # ---------- Freezing helpers ----------
    def _named_linear_modules(self):
        """Return list of (name, module) for all nn.Linear found (in order of appearance)."""
        return [(name, m) for name, m in self.model.named_modules() if isinstance(m, nn.Linear)]

    def _apply_freezing_policy(self):
        # Start by enabling everything
        for p in self.model.parameters():
            p.requires_grad = True

        # Explicit whitelist beats everything else
        if self.cfg.train_param_names:
            names = set(self.cfg.train_param_names)
            for name, p in self.model.named_parameters():
                p.requires_grad = (name in names)
            return

        # Regex / explicit blacklist
        if self.cfg.freeze_param_names or self.cfg.freeze_regex:
            frozen = set(self.cfg.freeze_param_names)
            pattern = re.compile(self.cfg.freeze_regex) if self.cfg.freeze_regex else None
            for name, p in self.model.named_parameters():
                if name in frozen or (pattern and pattern.search(name)):
                    p.requires_grad = False

        # Convenience: train only the final readout (last Linear)
        if self.cfg.train_only_output:
            for p in self.model.parameters():
                p.requires_grad = False
            linears = self._named_linear_modules()
            if not linears:
                raise RuntimeError("No nn.Linear modules found to unfreeze for train_only_output.")
            last_name, last_lin = linears[-1]
            for p in last_lin.parameters():
                p.requires_grad = True
            print(f"[Freeze] training only output layer: {last_name}")
            return

        # Convenience: train last k Linear layers (e.g., k=2 = last hidden + output)
        if self.cfg.train_last_n_linears and self.cfg.train_last_n_linears > 0:
            for p in self.model.parameters():
                p.requires_grad = False
            linears = self._named_linear_modules()
            if len(linears) < self.cfg.train_last_n_linears:
                raise RuntimeError(f"Requested last {self.cfg.train_last_n_linears} linears, but model has only {len(linears)}.")
            chosen = linears[-self.cfg.train_last_n_linears:]
            names = [n for (n, _) in chosen]
            for _, lin in chosen:
                for p in lin.parameters():
                    p.requires_grad = True
            print(f"[Freeze] training last {self.cfg.train_last_n_linears} linear layers: {names}")
            return

    # ---------- Train/Val ----------
    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Training]") \
                   if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs else train_loader

        for inputs, labels in iterator:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(train_loader.dataset)

    def _validate_one_epoch(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        iterator = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Validation]") \
                   if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs else val_loader

        with torch.no_grad():
            for inputs, labels in iterator:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                predicted = torch.sign(outputs) if outputs.shape[-1] == 1 else torch.argmax(outputs, dim=-1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = (correct_predictions / total_predictions) * 100
        return epoch_loss, epoch_acc

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        fname = ('BEST_MODEL' if is_best else 'LAST_CHECKPOINT') + self.cfg.model_name + '.pt'
        filepath = os.path.join(self.cfg.save_dir, fname)
        torch.save(state, filepath)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print("Training started...")
        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_acc = self._validate_one_epoch(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
                print(f"Epoch {epoch+1}/{self.cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            self.save_checkpoint(is_best=False)

            if val_loss < self.best_val_loss:
                if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving the best model.")
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

        print("Training finished.")
        return self.history


def evaluate_accuracy(model, X, y):
    device = device_string()
    model.eval()
    with torch.no_grad():
        X = X.to(device); y = y.to(device)
        out = model(X)
    #     if out.shape[-1] == 1:
    #         pred = torch.sign(out.squeeze(-1))
    #         acc = (pred == y.float().sign()).float().mean().item()
    #     else:
    #         pred = out.argmax(dim=-1)
    #         acc = (pred == y.long()).float().mean().item()
    # return acc
    return (torch.sign(out) == y).float().mean().detach().cpu().item()

def evaluate_mse(model, X, y):
    device = device_string()
    model.eval()
    with torch.no_grad():
        X = X.to(device); y = y.to(device)
        out = model(X)
    return ((out - y) ** 2).mean().detach().cpu().item()