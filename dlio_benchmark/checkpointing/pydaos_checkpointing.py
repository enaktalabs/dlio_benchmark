import logging
import torch
from pydaos.torch import Checkpoint as DaosCheckpoint

from dlio_benchmark.checkpointing.base_checkpointing import BaseCheckpointing
from dlio_benchmark.utils.utility import Profile
from dlio_benchmark.utils.config import ConfigArguments

from dlio_benchmark.common.constants import MODULE_CHECKPOINT

dlp = Profile(MODULE_CHECKPOINT)


class PyDaosTorchCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if PyDaosTorchCheckpointing.__instance is None:
            logging.basicConfig(level=logging.INFO)
            PyDaosTorchCheckpointing.__instance = PyDaosTorchCheckpointing()
        return PyDaosTorchCheckpointing.__instance

    @dlp.log_init
    def __init__(self):
        super().__init__("pt")

        args = ConfigArguments.get_instance()
        prefix = args.checkpoint_folder
        pool = args.checkpoint_daos_pool
        cont = args.checkpoint_daos_cont
        chunk_size = args.checkpoint_daos_chunk_size
        chunks_limit = args.checkpoint_daos_chunks_limit

        logging.info(f"Checkpointing is set to DAOS pool: {pool}, container: {cont}, prefix: {prefix}, chunk_size: {chunk_size} and chunks_limit: {chunks_limit}")
        self.ckpt = DaosCheckpoint(pool, cont, prefix, transfer_chunk_size=chunk_size, chunks_limit=chunks_limit)

    @dlp.log
    def get_tensor(self, size):
        return torch.randint(high=1, size=(size,), dtype=torch.int8)

    @dlp.log
    def save_state(self, suffix, state):
        name = self.get_name(suffix)
        with self.ckpt.writer(name) as f:
            torch.save(state, f)

    @dlp.log
    def checkpoint(self, epoch, step_number):
        super().checkpoint(epoch, step_number)

    @dlp.log
    def finalize(self):
        super().finalize()
