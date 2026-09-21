"""
BNN training and evaluation.

Copyright (c) 2026 Madelyn Cruz and Daniel Forger
University of Michigan. All rights reserved.
"""
from collections.abc import Mapping
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from zanj import ZANJ

from bnbp5.bnn_intralayer import BNN
from bnbp5.mnist_spiketrain_sliding import SpikeTrainMNIST

logger = logging.getLogger(__name__)

def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)

class Trainer:
    """
    BNN/DNN trainer with explicit dataset selection. 
    """
    def __init__(self, CFG1, CFG2, pretrained='', subjects=None,
                 ctrs1=None, ctrs2=None, num_classes=2, *, dataset='mnist',
                 data_root=None, eeg_root=None, device=None,
                 train_dataset=None, val_dataset=None, test_dataset=None, download=False, seed=15,
                 num_workers=0, pin_memory=None, grad_clip_norm=1000.0,
                 validation_fraction=0.3, test_fraction=0.20, epoch_seconds=4.0, gap_seconds=0.0,
                 bandpass=(0.5, 60.0), trusted_checkpoint=True):
        if dataset not in ('mnist', 'anesthesia', 'provided', 'none'):
            raise ValueError("dataset must be mnist, anesthesia, provided, or none")
        
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        
        if (train_dataset is not None or val_dataset is not None) and dataset != 'provided':
            raise ValueError("Use dataset='provided' when passing datasets")
        
        self.CFG1, self.CFG2 = CFG1, CFG2
        self.subjects = list(['UM_7'] if subjects is None else subjects)
        self.ctrs1 = None if ctrs1 is None else list(ctrs1)
        self.ctrs2 = None if ctrs2 is None else list(ctrs2)
        self.num_classes, self.seed = num_classes, int(seed)
        
        self.dataset_name = dataset
        self.data_root = Path(data_root) if data_root is not None else Path(__file__).resolve().parents[1] / 'data'
        self.eeg_root = None if eeg_root is None else Path(eeg_root)
        self.eeg_options = dict(
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            epoch_seconds=epoch_seconds,
            gap_seconds=gap_seconds,
            bandpass=bandpass,
        )

        target_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.model = BNN(CFG1).to(target_device)
        self.num_workers = num_workers
        self.pin_memory = target_device.type == 'cuda' if pin_memory is None else bool(pin_memory)
        self.grad_clip_norm = float(grad_clip_norm)
        self._loader_cache = {}
        self._shuffle_generator = torch.Generator().manual_seed(self.seed)
        self._analysis_generator = torch.Generator().manual_seed(self.seed + 2)
        self._optimizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.last_epoch = None

        if pretrained:
            self.load_model_from_file(pretrained, trusted_checkpoint=trusted_checkpoint)
        if dataset == "provided":
            self.set_datasets(train_dataset, val_dataset, test_dataset)
        elif dataset == "mnist":
            self.load_mnist(download=download)
        elif dataset == "anesthesia":
            self.load_anest()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.CFG1.lr)
        return self._optimizer

    def set_datasets(self, train_dataset, val_dataset, test_dataset=None):
        for name, dataset in (
            ("training", train_dataset),
            ("validation", val_dataset),
        ):
            if dataset is None or len(dataset) == 0:
                raise ValueError(f"Provide a nonempty {name} dataset.")

        if test_dataset is not None and len(test_dataset) == 0:
            raise ValueError("The testing dataset must be nonempty.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self._loader_cache = {}


    def _loader(self, phase, *, shuffle=False):
        datasets = {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset,
        }
        if phase not in datasets:
            raise ValueError(f"Unknown dataset phase: {phase}")
        if shuffle and phase != "train":
            raise ValueError("Only training batches may be shuffled.")

        dataset = datasets[phase]
        if dataset is None or len(dataset) == 0:
            raise ValueError(f"No nonempty {phase} dataset is configured.")

        batch_size = (
            self.CFG1.train_batch_sz
            if shuffle else self.CFG1.test_batch_sz
        )
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive.")

        workers = self.num_workers
        pin = self.pin_memory
        key = (id(dataset), batch_size, shuffle, workers, pin)
        slot = (phase, shuffle)

        if slot not in self._loader_cache or self._loader_cache[slot][0] != key:
            # Evaluation never advances the training-shuffle generator.
            if shuffle:
                generator = self._shuffle_generator
            else:
                seed_offset = {"train": 1, "validation": 2, "test": 3}[phase]
                generator = torch.Generator().manual_seed(
                    self.seed + seed_offset
                )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=pin,
                persistent_workers=workers > 0,
                worker_init_fn=_seed_worker,
                generator=generator,
            )
            self._loader_cache[slot] = (key, loader)

        return self._loader_cache[slot][1]

    def load_mnist(self, *, download=False):
        from torchvision import datasets, transforms
        if self.CFG1.model_dims[0] != 784 or self.CFG1.model_dims[-1] != 10:
            raise ValueError("MNIST requires 784 input features and 10 output classes")
        raw_root = self.data_root / 'mnist_torch'
        prepared = []
        for train, phase, phase_seed in [(True, 'train', self.seed), (False, 'validation', self.seed + 1)]:
            raw = datasets.MNIST(str(raw_root), train=train, download=download, transform=transforms.ToTensor())
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(phase_seed)
                prepared.append(SpikeTrainMNIST(raw, phase, self.CFG2,
                                               cache_dir=self.data_root, cache_seed=phase_seed))
        self.set_datasets(*prepared)

    def load_anest(self):
        """
        Load anesthesia data.
        """
        from pathlib import Path

        import h5py
        import mne
        import pandas as pd
        import scipy.io
        from torch.utils.data import TensorDataset

        root = getattr(self, "eeg_root", None)
        if root is None:
            raise ValueError("Set eeg_root to the anesthesia dataset folder.")
        root = Path(root)

        subjects = list(self.subjects)
        if len(subjects) != 1:
            raise ValueError("Specify exactly one anesthesia subject.")

        subject = subjects[0]
        subject_rows = {
            "UM_4": 2,
            "UM_7": 3,
            "UM_8": 4,
            "UM_9": 5,
            "UM_12": 6,
            "UM_13": 7,
            "UM_14": 8,
            "UM_18": 9,
            "UM_21": 10,
        }
        if subject not in subject_rows:
            raise ValueError(f"Unsupported subject: {subject}")
        if self.num_classes not in (2, 5, 6):
            raise ValueError("num_classes must be 2, 5, or 6.")

        sfreq = 500
        montage = mne.channels.read_custom_montage(
            str(root / "EGI_ChannelLocations" / "GSN-HydroCel-128.sfp")
        )

        event_file = (
            root
            / "McDonnell_Events_Info_Summary_TB_10_27_16_DL090817_UManesthesia.xlsx"
        )

        table = pd.read_excel(event_file, header=0)
        row = subject_rows[subject]

        pairs = [(1, 2), (3, 4), (5, 6)]
        pairs += [(7, 11), (11, 12), (12, 13), (13, 15), (15, 16)]
        pairs += [
            (j, j + 1)
            for j in (16, 20, 22, 26, 28, 32, 34, 38, 40, 44, 46, 50, 52)
        ]

        events = []
        for event_id, (start_column, stop_column) in enumerate(pairs):
            onset = float(table.loc[row, start_column])
            duration = float(
                table.loc[row, stop_column] - table.loc[row, start_column]
            )
            if not np.isfinite(onset) or not np.isfinite(duration):
                raise ValueError(f"Nonfinite timing for event {event_id}.")

            events.append((int(onset), int(duration), event_id))

        mat_files = sorted((root / subject).glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No MAT files found in {root / subject}")

        chunks = []
        for path in mat_files:
            try:
                mat_data = scipy.io.loadmat(str(path))
                chunk = mat_data["EEG"][0, 0][15]
            except NotImplementedError:
                with h5py.File(path, "r") as handle:
                    chunk = handle["EEG"]["data"][:, :128].T

            chunk = np.asarray(chunk)
            if chunk.ndim != 2:
                raise ValueError(
                    f"{path.name}: expected [channels, samples], got {chunk.shape}"
                )
            if chunk.shape[0] != len(montage.ch_names):
                raise ValueError(
                    f"{path.name}: found {chunk.shape[0]} channels, "
                    f"but montage has {len(montage.ch_names)}."
                )
            chunks.append(chunk)

        # Preserve channel order and concatenate recordings along time.
        eeg = np.concatenate(chunks, axis=1)
        del chunks

        info = mne.create_info(
            ch_names=montage.ch_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(eeg, info, verbose=False)
        raw.set_montage(montage)
        #raw = raw.copy().filter(self.eeg_options.bandpass,verbose=False)
        del eeg

        # Read settings from the revised Trainer when available.
        options = getattr(self, "eeg_options", {}) or {}
        validation_fraction = float(
            options.get("validation_fraction", 0.30)
        )
        gap_seconds = float(options.get("gap_seconds", 4.0))

        if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")

        if not np.isfinite(gap_seconds) or gap_seconds < 0:
            raise ValueError("gap_seconds must be finite and nonnegative.")

        gap_samples = int(round(gap_seconds * sfreq))

        # No bandpass filter: it was commented out in the original.
        tmin = -0.200
        tmax = 4.20
        first_offset = int(round(tmin * sfreq))  # -100
        last_offset = int(round(tmax * sfreq))  # 2100
        epoch_samples = last_offset - first_offset + 1  # 2201
        
        def map_label(event_id):
            """Preserve the original class definitions."""
            if self.num_classes == 2:
                if event_id <= 2:
                    return 0
                if event_id == 4:
                    return 1
                return None

            if event_id <= 2:
                return 0
            if event_id == 3:
                return 1
            if event_id == 4:
                return 2
            if event_id in (5, 6):
                return 3
            if event_id == 7:
                return 4
            return 5 if self.num_classes == 6 else None

        options = getattr(self, "eeg_options", {}) or {}
        validation_fraction = float(options.get("validation_fraction", 0.30))
        test_fraction = float(options.get("test_fraction", 0.20))
        gap_seconds = float(options.get("gap_seconds", 0.0))

        if (
            not np.isfinite(validation_fraction)
            or not np.isfinite(test_fraction)
            or validation_fraction <= 0
            or test_fraction <= 0
            or validation_fraction + test_fraction >= 1
        ):
            raise ValueError(
                "Validation and test fractions must be positive "
                "and their sum must be less than 1."
            )
        if not np.isfinite(gap_seconds) or gap_seconds < 0:
            raise ValueError("gap_seconds must be finite and nonnegative.")

        train_fraction = 1.0 - validation_fraction - test_fraction
        gap_samples = int(round(gap_seconds * sfreq))

        plans = {"train": [], "validation": [], "test": []}
        partitions = []
        previous_stop = 0

        left_margin = gap_samples // 2
        right_margin = gap_samples - left_margin

        for start, duration, event_id in sorted(events, key=lambda event: event[0]):
            label = map_label(event_id)
            if label is None:
                continue

            stop = start + duration
            if start < 0 or duration <= 0 or stop > raw.n_times:
                raise ValueError(
                    f"Event {event_id} has invalid bounds [{start}, {stop})."
                )
            if start < previous_stop:
                raise ValueError(f"Event {event_id} overlaps another selected event.")
            previous_stop = stop

            # Outer margins separate the end of one event from the next.
            left = start + left_margin
            right = stop - right_margin

            # Reserve two internal gaps: train/validation and validation/test.
            usable = right - left - 2 * gap_samples
            if usable <= 0:
                raise ValueError(f"Event {event_id} is too short for these gaps.")

            train_length = int(usable * train_fraction)
            val_length = int(usable * validation_fraction)
            test_length = usable - train_length - val_length

            if min(train_length, val_length, test_length) < epoch_samples:
                raise ValueError(
                    f"Event {event_id} cannot fit one complete epoch in each "
                    "of training, validation, and testing. Reduce the gap "
                    "or revise the split fractions."
                )

            train_stop = left + train_length
            val_start = train_stop + gap_samples
            val_stop = val_start + val_length
            test_start = val_stop + gap_samples

            bounds = {
                "train": (left, train_stop),
                "validation": (val_start, val_stop),
                "test": (test_start, right),
            }

            partitions.append({
                "event_id": int(event_id),
                "class": int(label),
                **{
                    phase: [int(first), int(last)]
                    for phase, (first, last) in bounds.items()
                },
            })

            for phase, (first, last) in bounds.items():
                for window_start in range(
                    first, last - epoch_samples + 1, epoch_samples
                ):
                    plans[phase].append((
                        window_start,
                        window_start + epoch_samples,
                        label,
                    ))

        seed = int(getattr(self, "seed", 15))
        train_seed, val_seed, test_seed = np.random.SeedSequence(seed).spawn(3)

        train_rng = np.random.default_rng(train_seed)
        val_rng = np.random.default_rng(val_seed)
        test_rng = np.random.default_rng(test_seed)

        train_plan = np.asarray(plans["train"], dtype=np.int64)
        val_plan = np.asarray(plans["validation"], dtype=np.int64)
        test_plan = np.asarray(plans["test"], dtype=np.int64)

        n_train = int(self.CFG2.n_samples_train)
        n_val = int(self.CFG2.n_samples_val)
        n_test = int(self.CFG2.n_samples_test)
        n_classes = self.num_classes
        
        groups = [
            np.flatnonzero(train_plan[:, 2] == label)
            for label in range(n_classes)
        ]

        per_class = min(
            n_train // n_classes,
            min(len(group) for group in groups),
        )
        selected = []

        print("train_plan", train_plan[:,2], len(train_plan[:,2]))
        
        selected = np.concatenate([
            train_rng.choice(group, size=per_class, replace=False)
            for group in groups
        ])
        train_plan = train_plan[train_rng.permutation(selected)]

        print(
            f"Selected {len(train_plan)} training epochs "
            f"({per_class} per class); requested {n_train}."
        )

        val_plan = val_plan[
            val_rng.choice(len(val_plan), size=min(n_val,len(val_plan)), replace=False)
        ]
        test_plan = test_plan[
            test_rng.choice(len(test_plan), size=min(n_test,len(test_plan)), replace=False)
        ]
        
        def materialize(plan):
            """Extract only selected, nonoverlapping epochs."""
            values = np.empty(
                (len(plan), epoch_samples, len(raw.ch_names)),
                dtype=np.float64,
            )
            for index, (first, last, _) in enumerate(plan):
                values[index] = raw.get_data(
                    start=int(first), stop=int(last)
                ).T

            if not np.isfinite(values).all():
                raise ValueError("EEG contains nonfinite values.")

            targets = torch.nn.functional.one_hot(
                torch.from_numpy(plan[:, 2].copy()),
                num_classes=self.num_classes,
            ).float()
            return torch.from_numpy(values), targets

        x_train, y_train = materialize(train_plan)
        x_val, y_val = materialize(val_plan)
        x_test, y_test = materialize(test_plan)
        del raw

        scale = float(x_train.abs().max().item())
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Invalid training maximum absolute amplitude.")

        x_train = x_train / scale
        x_val = x_val / scale
        x_test = x_test / scale

        self.set_datasets(
            TensorDataset(x_train, y_train),
            TensorDataset(x_val, y_val),
            TensorDataset(x_test, y_test),
        )

        print("Requested train/validation:", n_train, n_val)
        print("Actual train/validation:", len(x_train), len(x_val))
        print("Training class counts:",
            torch.bincount(y_train.argmax(1), minlength=n_classes).tolist())
        print("Validation class counts:",
            torch.bincount(y_val.argmax(1), minlength=n_classes).tolist())
        print("Testing class counts:",
            torch.bincount(y_test.argmax(1), minlength=n_classes).tolist())
        print("Gap in seconds:", gap_seconds)
        
        self.set_datasets(
            TensorDataset(x_train, y_train),
            TensorDataset(x_val, y_val),
            TensorDataset(x_test, y_test),
        )

    def _batch(self, batch, expected):
        non_blocking = self.device.type == 'cuda'
        return (batch[:, :, :self.CFG1.model_dims[0]].to(self.device, dtype=torch.float32, non_blocking=non_blocking),
                expected.to(self.device, dtype=torch.float32, non_blocking=non_blocking))

    def _loss(self, output, expected):
        return nn.functional.mse_loss(output, expected)

    def train(self, epoch=0, batches_val=-1, *,
              validate_train=False, validate_at_end=True):
        """
        Training on one epoch at with evaluation at the end of the epoch and when batches_val is positive.
        Returns accuracies, losses.
        """
        if not isinstance(batches_val, int) or batches_val == 0 or batches_val < -1:
            raise ValueError("batches_val must be -1 or a positive integer")

        loader = self._loader('train', shuffle=True)
        self.model.train()
        optimizer = self.optimizer
        losses = torch.empty(len(loader), device=self.device)
        total_loss = torch.zeros((), device=self.device)
        samples, accuracies, last_validation = 0, [], 0
        start = time.perf_counter()

        for batch_index, (batch, expected) in enumerate(loader, 1):
            print(batch_index)
            batch, expected = self._batch(batch, expected)
            optimizer.zero_grad(set_to_none=True)
            try:
                output = self.model(batch).mean(dim=1)
                loss = self._loss(output, expected)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm,
                                        error_if_nonfinite=True)
            except Exception:
                optimizer.zero_grad(set_to_none=True)
                raise
            
            optimizer.step()
            losses[batch_index - 1] = loss.detach()
            total_loss += loss.detach() * len(batch)
            samples += len(batch)

            if batches_val > 0 and batch_index % batches_val == 0:
                if validate_train:
                    logger.info("Training accuracy %.2f%%", self.validate(epoch, batch_index, False))
                accuracies.append(self.validate(epoch, batch_index))
                last_validation = batch_index
        if validate_at_end and last_validation != len(loader):
            if validate_train:
                logger.info("Training accuracy %.2f%%", self.validate(epoch, len(loader), False))
            accuracies.append(self.validate(epoch, len(loader)))
        loss_record = losses.cpu().tolist()
        self.last_epoch = dict(epoch=epoch, batches=len(loader), samples=samples,
                               mean_loss=total_loss.item() / samples,
                               elapsed_seconds=time.perf_counter() - start,
                               validation_accuracy=accuracies[-1] if accuracies else None)
        logger.info("Epoch %s: %s", epoch, self.last_epoch)
        return accuracies, loss_record

    def _evaluate(self, phase):
        loader = self._loader(phase)
        modes = [(module, module.training) for module in self.model.modules()]
        n_hit = torch.zeros((), dtype=torch.long, device=self.device)
        n_total = 0

        self.model.eval()
        try:
            with torch.no_grad():
                for batch, expected in loader:
                    batch, expected = self._batch(batch, expected)
                    output = self.model(batch).mean(dim=1)

                    n_hit += (
                        output.argmax(dim=1) == expected.argmax(dim=1)
                    ).sum()
                    n_total += len(batch)
        finally:
            for module, mode in modes:
                module.training = mode

        return 100.0 * n_hit.item() / n_total


    def validate(self, epoch=0, batch_idx=-1, use_val_dataset=True):
        phase = "validation" if use_val_dataset else "train"
        return self._evaluate(phase)


    def test(self):
        return self._evaluate("test")

    def load_model_from_file(self, pretrained, *, restore_optimizer=False, trusted_checkpoint=False):
        """
        Read a plain state_dict or checkpoint bundle
        """
        state = torch.load(pretrained, map_location=self.device, weights_only=not trusted_checkpoint)
        weights = state.get('model_state_dict', state.get('state_dict', state))

        self.model.load_state_dict(weights, strict=True)
        if restore_optimizer:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            # Adam moments from a different set of weights must not be reused.
            self._optimizer = None
        
        return state

    def measure_sliding_gradients(self, window_size, filename, stride=-1, *, include_diagnostics=False):
        """Measure derivatives without optimizer steps or parameter-gradient mutation.

        Main output keys are compatible with HebbianLearning.ipynb. Expensive
        per-window dL/d(avgs) matrices are optional, detached CPU snapshots.
        """
        if stride == -1:
            stride = window_size
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride must be a positive integer")
        if len(self.model.Ws) != 2:
            raise ValueError("Sliding-gradient export expects two weight layers")
        model_device = next(self.model.parameters()).device
        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("Gradient measurement requires a nonempty dataset")
        idx = int(torch.randint(0, len(self.train_dataset), (),
                               generator=getattr(self, '_analysis_generator', None)))
        sample, target = self.train_dataset[idx]
        sample = sample[:, :self.CFG1.model_dims[0]].unsqueeze(0).to(model_device)
        target = target.to(model_device)
        if window_size > sample.shape[1]:
            raise ValueError("window_size exceeds the number of timesteps")

        T2_out, interm_out = self.model(sample, include_intermediates=True)
        T2_out = T2_out.squeeze(0)
        # unfold returns a view [window, output, window_size].
        avgs = T2_out.unfold(0, window_size, stride).mean(dim=-1)
        window_cnt = avgs.shape[0]
        weights = tuple(layer.weight for layer in self.model.Ws)
        buffers = [torch.empty((window_cnt, weight.numel()), dtype=weight.dtype) for weight in weights]
        losses = []
        partialavs = []
        for i in range(window_cnt):
            loss = torch.nn.functional.mse_loss(avgs[i], target)
            requested = weights + ((avgs,) if include_diagnostics else ())
            grads = torch.autograd.grad(loss, requested, retain_graph=i + 1 < window_cnt)
            for buffer, grad in zip(buffers, grads[:2]):
                buffer[i].copy_(grad.detach().reshape(-1).cpu())
            losses.append(loss.detach().cpu())
            partialavs.append(grads[2].detach().cpu().clone() if include_diagnostics else None)

        # Weights do not change during this measurement. Reuse one CPU snapshot.
        snapshots = [weight.detach().cpu().clone() for weight in weights]
        ZANJ().save(
            dict(
                losses=losses, avgs=avgs.detach().cpu(), partialavs=partialavs,
                W1s=[snapshots[0]] * window_cnt, W2s=[snapshots[1]] * window_cnt,
                idx=idx, sample=sample.detach().cpu(), target=target.detach().cpu(),
                window_cnt=window_cnt, T2_out=T2_out.detach().cpu(),
                interm_out=[tuple(value.detach().cpu() for value in pair) for pair in interm_out],
                sliding_grad_1=buffers[0], sliding_grad_2=buffers[1],
                window_size=window_size, stride=stride, dt=self.CFG1.dt,
                include_diagnostics=include_diagnostics,
            ),
            filename,
        )