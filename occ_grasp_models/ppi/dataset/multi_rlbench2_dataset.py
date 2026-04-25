from typing import Any, Dict, Mapping, Sequence, Tuple

import hydra
import torch
from omegaconf import OmegaConf

from ppi.dataset.base_dataset import BaseDataset
from ppi.model.common.normalizer import LinearNormalizer


class MultiRLBench2Dataset(BaseDataset):
    def __init__(
        self,
        datasets,
        stats_filepath,
        sample_mode="task_uniform",
        samples_per_task=None,
        **kwargs,
    ):
        super().__init__()

        self.stats_filepath = stats_filepath
        self.sample_mode = sample_mode
        self.shared_kwargs = dict(kwargs)
        self.datasets = self._build_dataset_dict(datasets)
        self.task_names = tuple(self.datasets.keys())
        self.dataset_list = [self.datasets[name] for name in self.task_names]
        self.dataset_lengths = tuple(len(dataset) for dataset in self.dataset_list)

        if len(self.dataset_list) == 0:
            raise ValueError("MultiRLBench2Dataset requires at least one sub-dataset.")
        if any(length <= 0 for length in self.dataset_lengths):
            invalid = {
                name: length
                for name, length in zip(self.task_names, self.dataset_lengths)
                if length <= 0
            }
            raise ValueError(f"All sub-datasets must be non-empty, got {invalid}.")
        if self.sample_mode != "task_uniform":
            raise ValueError(
                f"Unsupported sample_mode '{self.sample_mode}'. "
                "Only 'task_uniform' is implemented."
            )

        if samples_per_task is None:
            # Match each task to the longest sub-dataset so task-uniform sampling
            # does not silently starve the larger tasks.
            samples_per_task = max(self.dataset_lengths)
        self.samples_per_task = int(samples_per_task)
        if self.samples_per_task <= 0:
            raise ValueError(
                f"samples_per_task must be positive, got {self.samples_per_task}."
            )

        self.num_tasks = len(self.dataset_list)
        self._length = self.num_tasks * self.samples_per_task

    def _build_dataset_dict(self, datasets) -> Dict[str, BaseDataset]:
        if isinstance(datasets, Mapping):
            dataset_items = list(datasets.items())
        elif isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
            dataset_items = [(str(idx), dataset) for idx, dataset in enumerate(datasets)]
        else:
            raise TypeError(
                "datasets must be a mapping or sequence of dataset configs/instances."
            )

        built_datasets = dict()
        for name, dataset in dataset_items:
            instance = self._instantiate_dataset(dataset)
            if not isinstance(instance, BaseDataset):
                raise TypeError(
                    f"Sub-dataset '{name}' must inherit BaseDataset, got {type(instance)}."
                )
            built_datasets[str(name)] = instance
        return built_datasets

    @staticmethod
    def _instantiate_dataset(dataset: Any) -> BaseDataset:
        if isinstance(dataset, BaseDataset):
            return dataset

        if OmegaConf.is_config(dataset):
            return hydra.utils.instantiate(dataset)

        if isinstance(dataset, Mapping) and "_target_" in dataset:
            return hydra.utils.instantiate(dataset)

        raise TypeError(
            "Each sub-dataset must be a BaseDataset instance or a Hydra config with "
            "a '_target_' field."
        )

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}.")

        task_idx = idx % self.num_tasks
        sample_round = idx // self.num_tasks
        sample_idx = sample_round % self.dataset_lengths[task_idx]
        return task_idx, sample_idx

    def _get_actions_from_dataset(self, dataset: BaseDataset) -> torch.Tensor:
        if hasattr(dataset, "replay_buffer"):
            replay_buffer = getattr(dataset, "replay_buffer")
            try:
                return torch.as_tensor(replay_buffer["action"])
            except Exception:
                pass

        try:
            actions = dataset.get_all_actions()
        except NotImplementedError as exc:
            raise AttributeError(
                f"Sub-dataset '{type(dataset).__name__}' does not expose actions for "
                "normalizer fitting."
            ) from exc

        return torch.as_tensor(actions)

    def get_validation_dataset(self) -> "MultiRLBench2Dataset":
        val_datasets = {
            name: dataset.get_validation_dataset()
            for name, dataset in self.datasets.items()
        }
        return MultiRLBench2Dataset(
            datasets=val_datasets,
            stats_filepath=self.stats_filepath,
            sample_mode=self.sample_mode,
            **self.shared_kwargs,
        )

    def get_normalizer(self, mode="limits", **kwargs) -> LinearNormalizer:
        if not self.stats_filepath:
            raise ValueError("stats_filepath must be set for MultiRLBench2Dataset.")

        try:
            state_dict = torch.load(
                self.stats_filepath, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(self.stats_filepath, map_location="cpu")
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(state_dict)
        normalizer.fit(
            data={"action": self.get_all_actions()},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = [
            self._get_actions_from_dataset(dataset) for dataset in self.dataset_list
        ]
        return torch.cat(actions, dim=0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        task_idx, sample_idx = self._resolve_index(idx)
        return self.dataset_list[task_idx][sample_idx]
