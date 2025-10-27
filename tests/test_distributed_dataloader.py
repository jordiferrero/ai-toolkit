import os
import importlib.machinery
import os
import sys
import types
import unittest
from unittest import mock

from torch.utils.data import Dataset

if "torchaudio" not in sys.modules:
    torchaudio_stub = types.ModuleType("torchaudio")
    torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = torchaudio_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2_stub

if "albumentations" not in sys.modules:
    albumentations_stub = types.ModuleType("albumentations")
    albumentations_stub.__spec__ = importlib.machinery.ModuleSpec("albumentations", loader=None)
    sys.modules["albumentations"] = albumentations_stub

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch


class _DummyAccelerator:
    def __init__(self, num_processes=2, process_index=0):
        self.num_processes = num_processes
        self.process_index = process_index


class _FakeDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, batch_size=1, sd=None):
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self._data = list(range(12))

    def __len__(self):
        if self.dataset_config.buckets:
            bucket_size = max(self.batch_size, 1)
            return max(len(self._data) // bucket_size, 1)
        return len(self._data)

    def __getitem__(self, index):
        if self.dataset_config.buckets:
            bucket_size = max(self.batch_size, 1)
            start = index * bucket_size
            end = start + bucket_size
            return [self._data[i % len(self._data)] for i in range(start, end)]
        return self._data[index]


class DistributedDataLoaderTests(unittest.TestCase):
    def _patch_accelerator(self, num_processes=2, process_index=0):
        return mock.patch(
            "toolkit.data_loader.get_accelerator",
            return_value=_DummyAccelerator(num_processes=num_processes, process_index=process_index),
        )

    def _patch_dataset(self):
        return mock.patch("toolkit.data_loader.AiToolkitDataset", _FakeDataset)

    def test_injects_distributed_sampler_for_multiple_processes(self):
        dataset_config = DatasetConfig(
            dataset_path="dummy",
            buckets=False,
            num_workers=0,
            prefetch_factor=2,
        )
        with self._patch_dataset(), self._patch_accelerator():
            dataloader = get_dataloader_from_datasets([dataset_config], batch_size=3, sd=None)
        sampler = dataloader.sampler
        self.assertIsNotNone(sampler)
        self.assertTrue(hasattr(sampler, "num_replicas"))
        self.assertEqual(sampler.num_replicas, 2)

    def test_trigger_dataloader_setup_epoch_updates_sampler_epoch(self):
        dataset_config = DatasetConfig(
            dataset_path="dummy",
            buckets=False,
            num_workers=0,
            prefetch_factor=2,
        )
        with self._patch_dataset(), self._patch_accelerator():
            dataloader = get_dataloader_from_datasets([dataset_config], batch_size=2, sd=None)

        self.assertFalse(hasattr(dataloader, "_aitk_sampler_epoch"))
        trigger_dataloader_setup_epoch(dataloader)
        self.assertEqual(dataloader._aitk_sampler_epoch, 0)
        self.assertEqual(getattr(dataloader.sampler, "epoch", None), 0)

        trigger_dataloader_setup_epoch(dataloader)
        self.assertEqual(dataloader._aitk_sampler_epoch, 1)
        self.assertEqual(getattr(dataloader.sampler, "epoch", None), 1)

    def test_bucketed_dataset_uses_sampler(self):
        dataset_config = DatasetConfig(
            dataset_path="dummy",
            buckets=True,
            num_workers=0,
            prefetch_factor=2,
        )
        with self._patch_dataset(), self._patch_accelerator():
            dataloader = get_dataloader_from_datasets([dataset_config], batch_size=4, sd=None)
        sampler = dataloader.sampler
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.num_replicas, 2)
        self.assertIsNone(dataloader.batch_size)


if __name__ == "__main__":
    unittest.main()
