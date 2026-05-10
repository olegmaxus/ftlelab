import torch
import torch.nn as nn
import os, re
from dataclasses import asdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import device_string
from .config import TrainConfig, LOSS_MAP, OPTIMIZER_MAP, PARAM_MODULES
from .metrics import (
    binary_accuracy,
    multiclass_accuracy,
    mse_metric,
    reconstruction_error,
    vae_kl_divergence,
)


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.device = device_string()
        self.model.to(self.device)

        print(f"Trainer for MODEL_{self.cfg.model_name} initialized on device: {self.device}")

        loss_name = self.cfg.loss.lower()
        loss_class = LOSS_MAP.get(loss_name)
        if not loss_class:
            raise ValueError(f"Unknown loss: '{self.cfg.loss}'. Supported: {list(LOSS_MAP.keys())}")
        self.criterion = loss_class()

        self._apply_freezing_policy()

        optimizer_name = self.cfg.optimizer.lower()
        opt_cls = OPTIMIZER_MAP.get(optimizer_name)
        if not opt_cls:
            raise ValueError(f"Unknown optimizer: '{self.cfg.optimizer}'. Supported: {list(OPTIMIZER_MAP.keys())}")

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable parameters after applying the freezing policy.")

        if optimizer_name == "sgd":
            self.optimizer = opt_cls(
                params, lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        else:
            self.optimizer = opt_cls(
                params, lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": [],
            "val_metric_name": self._metric_name(),
        }

        if self.cfg.task == "vae":
            self.history["val_recon_loss"] = []
            self.history["val_kl_loss"] = []

        os.makedirs(self.cfg.save_dir, exist_ok=True)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    # ============================================================
    # Freezing helpers
    # ============================================================

    def _named_param_modules(self):
        return [(name, m) for name, m in self.model.named_modules() if isinstance(m, PARAM_MODULES)]

    def _set_module_trainable(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _apply_freezing_policy(self):
        for p in self.model.parameters():
            p.requires_grad = True

        # explicit parameter whitelist
        if self.cfg.train_param_names:
            names = set(self.cfg.train_param_names)
            for name, p in self.model.named_parameters():
                p.requires_grad = (name in names)
            return

        # freeze by module name
        if self.cfg.freeze_module_names:
            named_modules = dict(self.model.named_modules())
            for name in self.cfg.freeze_module_names:
                if name not in named_modules:
                    raise ValueError(f"Module '{name}' not found in model.")
                self._set_module_trainable(named_modules[name], False)

        # train only selected modules
        if self.cfg.train_module_names:
            for p in self.model.parameters():
                p.requires_grad = False
            named_modules = dict(self.model.named_modules())
            for name in self.cfg.train_module_names:
                if name not in named_modules:
                    raise ValueError(f"Module '{name}' not found in model.")
                self._set_module_trainable(named_modules[name], True)
            print(f"[Freeze] training only modules: {self.cfg.train_module_names}")
            return

        # explicit blacklist / regex
        if self.cfg.freeze_param_names or self.cfg.freeze_regex:
            frozen = set(self.cfg.freeze_param_names)
            pattern = re.compile(self.cfg.freeze_regex) if self.cfg.freeze_regex else None
            for name, p in self.model.named_parameters():
                if name in frozen or (pattern and pattern.search(name)):
                    p.requires_grad = False

        # convenience: final parameterized module
        if self.cfg.train_only_output:
            for p in self.model.parameters():
                p.requires_grad = False
            mods = self._named_param_modules()
            if not mods:
                raise RuntimeError("No parameterized modules found.")
            last_name, last_mod = mods[-1]
            self._set_module_trainable(last_mod, True)
            print(f"[Freeze] training only output module: {last_name}")
            return

        # convenience: last n parameterized modules
        if self.cfg.train_last_n_param_modules > 0:
            for p in self.model.parameters():
                p.requires_grad = False
            mods = self._named_param_modules()
            if len(mods) < self.cfg.train_last_n_param_modules:
                raise RuntimeError(
                    f"Requested last {self.cfg.train_last_n_param_modules} parameterized modules, "
                    f"but model has only {len(mods)}."
                )
            chosen = mods[-self.cfg.train_last_n_param_modules:]
            for _, mod in chosen:
                self._set_module_trainable(mod, True)
            print(f"[Freeze] training last {self.cfg.train_last_n_param_modules} modules: {[n for n, _ in chosen]}")
            return

    # ============================================================
    # Task helpers
    # ============================================================

    def _metric_name(self):
        if self.cfg.task in {"binary", "multiclass"}:
            return "accuracy"
        if self.cfg.task in {"autoencoder", "vae"}:
            return "reconstruction"
        return "metric"

    def _unpack_batch(self, batch):
        """
        Supports:
          - (x, y)
          - (x,)
          - x
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 1:
                x, y = batch[0], None
            else:
                raise ValueError("Batch should be x, (x,), or (x, y).")
        else:
            x, y = batch, None
        return x.to(self.device), None if y is None else y.to(self.device)
    
    def _compute_loss_and_metric(self, outputs, targets, inputs):
        """
        Returns:
            loss, metric_value, extra_dict
        """

        # ---------------- binary classification ----------------
        if self.cfg.task == "binary":
            logits = outputs

            if self.cfg.loss == "bce_logits":
                y = targets.float()
                loss = self.criterion(logits.squeeze(-1), y.squeeze(-1))
                metric = binary_accuracy(logits.squeeze(-1), y.squeeze(-1), mode="bce_logits")

            elif self.cfg.loss == "bce":
                y = targets.float()
                loss = self.criterion(logits.squeeze(-1), y.squeeze(-1))
                metric = binary_accuracy(logits.squeeze(-1), y.squeeze(-1), mode="bce")

            elif self.cfg.loss == "mse":
                y = targets.float()
                loss = self.criterion(logits, y)
                metric = binary_accuracy(logits, y, mode="mse")

            else:
                raise ValueError("For binary task use loss='mse', 'bce', or 'bce_logits'.")

            return loss, metric, {}

        # ---------------- multiclass classification ----------------
        if self.cfg.task == "multiclass":
            logits = outputs
            y = targets.long().view(-1)
            loss = self.criterion(logits, y)
            metric = multiclass_accuracy(logits, y)
            return loss, metric, {}

        # ---------------- autoencoder ----------------
        if self.cfg.task == "autoencoder":
            recon = outputs[0] if isinstance(outputs, tuple) else outputs
            target = inputs if self.cfg.target_from_input or targets is None else targets
            loss = self.criterion(recon, target)
            metric = reconstruction_error(recon, target)
            return loss, metric, {}

        # ---------------- VAE ----------------
        if self.cfg.task == "vae":
            if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
                raise ValueError("VAE task expects model to return (recon, mu, logvar).")

            recon, mu, logvar = outputs[:3]
            target = inputs if self.cfg.target_from_input or targets is None else targets

            recon_loss = self.criterion(recon, target)
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )
            loss = recon_loss + self.cfg.beta * kl_loss
            metric = reconstruction_error(recon, target)

            return loss, metric, {
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
            }

        raise ValueError(f"Unknown task: {self.cfg.task}")

    # def _compute_loss_and_metric(self, outputs, targets, inputs):
    #     """
    #     Returns:
    #         loss, metric_value, extra_dict
    #     """

    #     # ---------------- binary classification ----------------
    #     if self.cfg.task == "binary":
    #         logits = outputs

    #         if self.cfg.loss == "bce_logits":
    #             y = targets.float()
    #             loss = self.criterion(logits.squeeze(-1), y.squeeze(-1))
    #             pred = (logits.squeeze(-1) >= 0).float()
    #             metric = (pred == y.squeeze(-1)).float().mean().item()

    #         elif self.cfg.loss == "mse":
    #             y = targets.float()
    #             loss = self.criterion(logits, y)
    #             pred = torch.sign(logits)
    #             metric = (pred == torch.sign(y)).float().mean().item()

    #         else:
    #             raise ValueError("For binary task use loss='mse' or 'bce_logits'.")

    #         return loss, metric, {}

    #     # ---------------- multiclass classification ----------------
    #     if self.cfg.task == "multiclass":
    #         logits = outputs
    #         y = targets.long().view(-1)
    #         loss = self.criterion(logits, y)
    #         pred = logits.argmax(dim=-1)
    #         metric = (pred == y).float().mean().item()
    #         return loss, metric, {}

    #     # ---------------- autoencoder ----------------
    #     if self.cfg.task == "autoencoder":
    #         recon = outputs[0] if isinstance(outputs, tuple) else outputs
    #         target = inputs if self.cfg.target_from_input or targets is None else targets
    #         loss = self.criterion(recon, target)
    #         metric = loss.item()  # reconstruction loss
    #         return loss, metric, {}

    #     # ---------------- VAE ----------------
    #     if self.cfg.task == "vae":
    #         if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
    #             raise ValueError("VAE task expects model to return (recon, mu, logvar).")

    #         recon, mu, logvar = outputs[:3]
    #         target = inputs if self.cfg.target_from_input or targets is None else targets

    #         recon_loss = self.criterion(recon, target)
    #         kl_loss = -0.5 * torch.mean(
    #             torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    #         )
    #         loss = recon_loss + self.cfg.beta * kl_loss
    #         metric = recon_loss.item()

    #         return loss, metric, {
    #             "recon_loss": recon_loss.item(),
    #             "kl_loss": kl_loss.item(),
    #         }

    #     raise ValueError(f"Unknown task: {self.cfg.task}")

    # ============================================================
    # Train / validate
    # ============================================================

    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        iterator = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Training]"
        ) if ((self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs) else train_loader

        for batch in iterator:
            inputs, targets = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss, _, _ = self._compute_loss_and_metric(outputs, targets, inputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return running_loss / len(train_loader.dataset)

    def _validate_one_epoch(self, val_loader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0

        extra_sums = {"recon_loss": 0.0, "kl_loss": 0.0}

        iterator = tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.cfg.epochs} [Validation]"
        ) if ((self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs) else val_loader

        with torch.no_grad():
            for batch in iterator:
                inputs, targets = self._unpack_batch(batch)
                outputs = self.model(inputs)

                loss, metric, extra = self._compute_loss_and_metric(outputs, targets, inputs)

                bs = inputs.size(0)
                running_loss += loss.item() * bs
                running_metric += metric * bs
                total_samples += bs

                for key in extra:
                    extra_sums[key] += extra[key] * bs

        epoch_loss = running_loss / total_samples
        epoch_metric = running_metric / total_samples

        extras_out = {k: v / total_samples for k, v in extra_sums.items() if total_samples > 0}
        return epoch_loss, epoch_metric, extras_out

    # ============================================================
    # Checkpoints
    # ============================================================

    def save_checkpoint(self, is_best=False):
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg),
        }

        fname = ("BEST_MODEL" if is_best else "LAST_CHECKPOINT") + self.cfg.model_name + ".pt"
        filepath = os.path.join(self.cfg.save_dir, fname)
        torch.save(state, filepath)

    # ============================================================
    # Main loop
    # ============================================================

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print("Training started...")

        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_metric, extras = self._validate_one_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            if self.cfg.task == "vae":
                self.history["val_recon_loss"].append(extras.get("recon_loss", float("nan")))
                self.history["val_kl_loss"].append(extras.get("kl_loss", float("nan")))

            if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
                msg = (
                    f"Epoch {epoch+1}/{self.cfg.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val {self.history['val_metric_name']}: {val_metric:.4f}"
                )
                if self.cfg.task == "vae":
                    msg += f" | Recon: {extras.get('recon_loss', float('nan')):.4f} | KL: {extras.get('kl_loss', float('nan')):.4f}"
                print(msg)

            self.save_checkpoint(is_best=False)

            if val_loss < self.best_val_loss:
                if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.epochs:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

        print("Training finished.")
        return self.history
    
    def evaluate(self, loader: DataLoader):
        """
        Generic evaluation on a loader using the current task/loss setup.
        """
        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0
        total_samples = 0

        extra_sums = {"recon_loss": 0.0, "kl_loss": 0.0}

        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack_batch(batch)
                outputs = self.model(inputs)

                loss, metric, extra = self._compute_loss_and_metric(outputs, targets, inputs)

                bs = inputs.size(0)
                running_loss += loss.item() * bs
                running_metric += metric * bs
                total_samples += bs

                for key in extra:
                    extra_sums[key] += extra[key] * bs

        result = {
            "loss": running_loss / total_samples,
            self.history["val_metric_name"]: running_metric / total_samples,
        }

        for key in extra_sums:
            if total_samples > 0:
                result[key] = extra_sums[key] / total_samples

        return result