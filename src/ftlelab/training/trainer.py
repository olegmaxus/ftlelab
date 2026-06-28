import os, re, json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import jax
from tqdm import tqdm
from dataclasses import asdict

from ..utils import device_string
from .config import TrainConfig, LOSS_MAP, OPTIMIZER_MAP, PARAM_MODULES
from ..ftle.jax_transfer import pytorch_dense_to_jax_params
from ..ftle.jax_core import ftle_field_batched_between
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
            "stop_reason": None,
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


    def _target_accuracy_reached(self, val_metric: float) -> bool:
        if self.cfg.task not in ("binary", "multiclass"):
            return False
        if self.cfg.target_val_acc is None:
            return False
        return val_metric >= float(self.cfg.target_val_acc)
    
    # ============================================================
    # Train / validate
    # ============================================================

    def _train_one_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        iterator = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.cfg.max_epochs} [Training]"
        ) if ((self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.max_epochs) else train_loader

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
            desc=f"Epoch {self.current_epoch+1}/{self.cfg.max_epochs} [Validation]"
        ) if ((self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.max_epochs) else val_loader

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
    # FTLEs
    # ============================================================

    def _ftle_enabled(self) -> bool:
        return bool(self.cfg.compute_ftle and self.cfg.ftle_config is not None)
    
    def _should_ftle_be_computed_this_epoch(self) -> bool:
        if not self._ftle_enabled():
            return False
        ep = self.current_epoch + 1
        start = max(1, int(self.cfg.ftle_start))
        return ep >= start and ((ep - start) % int(self.cfg.ftle_every) == 0)
    
    def _spec_to_name(self, spec):
        if spec == "input":
            return "input"
        if spec == "output":
            return "output"
        if isinstance(spec, tuple) and spec[0] == "hidden_k":
            return f"hidden-{int(spec[1])}"
        raise ValueError(f"usupported layer spec: {spec}")
    
    def _resolve_ftle_pairs(self):
        ftle_cfg = self.cfg.ftle_config

        # explicit transitions take priority
        if ftle_cfg.layer_pairs:
            return list(ftle_cfg.layer_pairs)
        
        # dense-only auto mode for now
        if ftle_cfg.model_type != "dense":
            raise NotImplementedError("Auto FTLE pair resolution currently supports only dense models.")

        if not hasattr(self.model, "hidden_depth"):
            raise RuntimeError("Model does not expose self.hidden_depth required for auto FTLE layer resolution.")

        hidden_depth = int(self.model.hidden_depth)

        # all input -> every hidden + output
        if ftle_cfg.layers == "all":
            pairs = [("input", ("hidden_k", k)) for k in range(1, hidden_depth + 1)]
            pairs.append(("input", "output"))
            return pairs

        # selected input -> layers
        pairs = []
        for item in ftle_cfg.layers:
            if item == "output":
                pairs.append(("input", "output"))
            elif isinstance(item, int):
                pairs.append(("input", ("hidden_k", item)))
            elif isinstance(item, tuple):
                pairs.append(("input", item))
            else:
                raise ValueError(f"Unsupported FTLE layer selector: {item}")
        return pairs
    

    def _compute_and_save_ftle_jax(self):
        ftle_cfg = self.cfg.ftle_config
        if ftle_cfg.grid_X is None:
            raise ValueError("ftle_config.grid_X must be provided for FTLE evaluation.")

        if ftle_cfg.save_format != "npy":
            raise ValueError("For the JAX backend, save_format must currently be 'npy'.")

        jax.config.update("jax_enable_x64", bool(ftle_cfg.enable_x64))

        self.model.eval()
        params = pytorch_dense_to_jax_params(self.model)
        pairs = self._resolve_ftle_pairs()

        epoch_num = self.current_epoch + 1
        epoch_dir = os.path.join(
            self.cfg.save_dir,
            ftle_cfg.save_subdir,
            self.cfg.model_name,
            f"epoch_{epoch_num:04d}",
        )
        os.makedirs(epoch_dir, exist_ok=True)

        snapshot = {
            "epoch": epoch_num,
            "backend": "jax",
            "files": {},
        }

        for start_spec, end_spec in pairs:
            arr = ftle_field_batched_between(
                model_type=ftle_cfg.model_type,
                params=params,
                X_np=np.asarray(ftle_cfg.grid_X),
                start_layer_spec=start_spec,
                end_layer_spec=end_spec,
                time_L=None,  # auto = layer distance
                batch_size=ftle_cfg.batch_size,
                activation=ftle_cfg.activation,
                output_activation=ftle_cfg.output_activation,
                exact_if_dim_le=ftle_cfg.exact_if_dim_le,
                max_steps=ftle_cfg.max_steps,
                tol=ftle_cfg.tol,
                dtype=ftle_cfg.dtype,
            )

            fname = f"{self._spec_to_name(start_spec)}__to__{self._spec_to_name(end_spec)}.npy"
            fpath = os.path.join(epoch_dir, fname)
            np.save(fpath, arr)
            snapshot["files"][f"{start_spec}->{end_spec}"] = fpath

        meta_path = os.path.join(epoch_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        self.history.setdefault("ftle_snapshots", []).append(snapshot)

    
    def _maybe_compute_ftle(self):
        if not self._should_ftle_be_computed_this_epoch():
            return
        if self.cfg.ftle_backend.lower() != "jax":
            raise NotImplementedError("Only Jax FTLE backend is implemented for now.")

        self._compute_and_save_ftle_jax()

    # ============================================================
    # Main loop
    # ============================================================

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        print("Training started...")

        for epoch in range(self.cfg.max_epochs):
            self.current_epoch = epoch

            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_metric, extras = self._validate_one_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            if self.cfg.task == "vae":
                self.history["val_recon_loss"].append(extras.get("recon_loss", float("nan")))
                self.history["val_kl_loss"].append(extras.get("kl_loss", float("nan")))

            if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.max_epochs:
                msg = (
                    f"Epoch {epoch+1}/{self.cfg.max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val {self.history['val_metric_name']}: {val_metric:.4f}"
                )
                if self.cfg.task == "vae":
                    msg += f" | Recon: {extras.get('recon_loss', float('nan')):.4f} | KL: {extras.get('kl_loss', float('nan')):.4f}"
                print(msg)

            self.save_checkpoint(is_best=False)

            if val_loss < self.best_val_loss:
                if (self.current_epoch + 1) % self.cfg.print_every == 0 or (self.current_epoch + 1) == self.cfg.max_epochs:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)

            # FTLE evaluation
            self._maybe_compute_ftle()

            # Stop if target accuracy is reached:
            if self._target_accuracy_reached(val_metric):
                print(
                    f"Target validation accuracy {self.cfg.target_val_acc:.4f} "
                    f"reached at epoch {self.current_epoch + 1} "
                    f"(got {val_metric:.4f})."
                )
                self.history["stop_reason"] = "target_val_acc_reached"
                break
        
        if self.history["stop_reason"] is None:
            self.history["stop_reason"] = "max_epochs_reached"

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