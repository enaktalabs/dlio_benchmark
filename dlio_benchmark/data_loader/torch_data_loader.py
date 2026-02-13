"""
   Copyright (c) 2025, UChicago Argonne, LLC
   Copyright (c) 2026, Enakta Labs, LTD
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import math
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import DatasetType, DataLoaderType, FormatType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI, Profile, dft_ai
from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.reader.npz_reader_daos import NPZReaderDaos

dlp = Profile(MODULE_DATA_LOADER)


class TorchDataset(Dataset):
    """
    Currently, we only support loading one sample per file
    TODO: support multiple samples per file
    """

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.logger = args.logger
        self.dlp_logger = None
        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        self.logger.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")
        self.reader = ReaderFactory.get_reader(type=self.format_type,
                                               dataset_type=self.dataset_type,
                                               thread_index=worker_id,
                                               epoch_number=self.epoch_number)

    def __del__(self):
        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return self.num_samples

    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.reader.read_index(image_idx, step)

class TorchDaosDataset(Dataset):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch, num_samples, num_workers, batch_size, pool, cont, prefix):
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch_number = epoch
        self.num_samples = num_samples
        self.reader = None
        self.num_images_read = 0
        self.batch_size = batch_size
        args = ConfigArguments.get_instance()
        self.serial_args = pickle.dumps(args)
        self.logger = args.logger
        self.dlp_logger = None

        # to avoid loading pydoas.torch at the top level if not needed or not installed
        from pydaos.torch import Dataset as DaosDataset
        self.dataset = DaosDataset(pool=pool, cont=cont, path=prefix)

        # DaosDataset enumerates files in dataset in parallel, so the order is non deterministic,
        # so index needs to be created of filename -> index inside the current dataset instance.
        # The index can be shared between workers as they share the same instance but have to be re-created for different ranks (which will happen automatically).
        self._dataset_index = {}
        for i, (path, _) in enumerate(self.dataset.objects):
            self._dataset_index[path] = i

        if num_workers == 0:
            self.worker_init(-1)

    @dlp.log
    def worker_init(self, worker_id):
        pickle.loads(self.serial_args)
        _args = ConfigArguments.get_instance()
        _args.configure_dlio_logging(is_child=True)
        self.dlp_logger = _args.configure_dftracer(is_child=True, use_pid=True)
        self.logger.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")

        if self.format_type != FormatType.NPZ:
            raise ValueError(f"unsupported format: {self.format_type}")

        # can't use ReaderFactory as it would require extend the interface with additional parameters
        self.reader = NPZReaderDaos(dataset_type=self.dataset_type,
                                    thread_index=worker_id,
                                    epoch=self.epoch_number,
                                    dataset=self.dataset,
                                    dataset_index=self._dataset_index)

        # still need to initialize underlying DaosDataset to share connection between workers
        self.dataset.worker_init(worker_id)

    def __del__(self):
        if self.dataset:
            del self.dataset

        if self.dlp_logger:
            self.dlp_logger.finalize()

    @dlp.log
    def __len__(self):
        return len(self.dataset) # should it be len of dataset of what num_samples passed in constructor ?

    def __getitem__(self, image_idx):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        self.logger.debug(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {image_idx} sample")
        dlp.update(step=step)
        dft_ai.update(step=step)
        return self.reader.read_index(image_idx, step)

#    def __getitems__(self, indices):
#        how to effectively use this optimization provided by DaosDataset ???

class dlio_sampler(Sampler):
    def __init__(self, rank, size, num_samples, epochs):
        self.size = size
        self.rank = rank
        self.num_samples = num_samples
        self.epochs = epochs
        samples_per_proc = int(math.ceil(num_samples/size)) 
        start_sample = self.rank * samples_per_proc
        end_sample = (self.rank + 1) * samples_per_proc - 1
        if end_sample > num_samples - 1:
            end_sample = num_samples - 1
        self.indices = list(range(start_sample, end_sample + 1))


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for sample in self.indices:
            yield sample


class TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number, data_loader_type):
        super().__init__(format_type, dataset_type, epoch_number, data_loader_type)

    @dlp.log
    def read(self):
        dataset = None
        if self.data_loader_type == DataLoaderType.PYTORCH:
            dataset = TorchDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                                   self._args.read_threads, self.batch_size)
        elif self.data_loader_type == DataLoaderType.DAOS_PYTORCH:

            prefix = os.path.join(self._args.data_folder, f"{self.dataset_type}")
            dataset = TorchDaosDataset(self.format_type, self.dataset_type, self.epoch_number, self.num_samples,
                                       self._args.read_threads, self.batch_size, pool=self._args.daos_pool,
                                       cont=self._args.daos_cont, prefix=prefix)
            self.num_samples = len(dataset)
        else:
            raise ValueError(f"Unsupported data loader type {self.data_loader_type}")

        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
        if self._args.read_threads >= 1:
            prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        else:
            prefetch_factor = self._args.prefetch_size
        if prefetch_factor > 0:
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        else:
            prefetch_factor = 2
            if self._args.my_rank == 0:
                self.logger.debug(
                    f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")
        self.logger.debug(f"{utcnow()} Setup dataloader with {self._args.read_threads} workers {torch.__version__}")
        if self._args.read_threads==0:
            kwargs={}
        else:
            kwargs={'multiprocessing_context':self._args.multiprocessing_context,
                    'prefetch_factor': prefetch_factor}
            if torch.__version__ != '1.3.1':       
                kwargs['persistent_workers'] = True
        if torch.__version__ == '1.3.1':
            if 'prefetch_factor' in kwargs:
                del kwargs['prefetch_factor']
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init, 
                                       **kwargs)
        else: 
            self._dataset = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       sampler=sampler,
                                       num_workers=self._args.read_threads,
                                       pin_memory=self._args.pin_memory,
                                       drop_last=True,
                                       worker_init_fn=dataset.worker_init,
                                       **kwargs)  # 2 is the default value
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataset) * self.batch_size} files")

        # self._dataset.sampler.set_epoch(epoch_number)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        self.logger.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        step = 1
        for batch in dft_ai.dataloader.fetch.iter(self._dataset):
            dlp.update(step=step)
            dft_ai.update(step=step)
            step += 1
            yield batch
        self.epoch_number += 1
        dlp.update(epoch=self.epoch_number)
        dft_ai.update(epoch=self.epoch_number)

    @dlp.log
    def finalize(self):
        pass
