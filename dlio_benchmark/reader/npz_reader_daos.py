import io
import numpy as np

from dlio_benchmark.common.constants import MODULE_DATA_READER
from dlio_benchmark.reader.reader_handler import FormatReader
from dlio_benchmark.utils.utility import Profile

dlp = Profile(MODULE_DATA_READER)


class NPZReaderDaos(FormatReader):
    """
    Reader for NPZ files
    """

    @dlp.log_init
    def __init__(self, dataset_type, thread_index, epoch, dataset, dataset_index):
        super().__init__(dataset_type, thread_index)
        self._dataset = dataset
        self._dataset_index = dataset_index

    @dlp.log
    def open(self, filename):
        super().open(filename)

        index = self._dataset_index[filename]
        sample = self._dataset[index]
        return np.load(io.BytesIO(sample), allow_pickle=True)['x']

    @dlp.log
    def close(self, filename):
        super().close(filename)

    @dlp.log
    def get_sample(self, filename, sample_index):
        super().get_sample(filename, sample_index)
        image = self.open_file_map[filename][..., sample_index]
        dlp.update(image_size=image.nbytes)

    def next(self):
        for batch in super().next():
            yield batch

    @dlp.log
    def read_index(self, image_idx, step):
        dlp.update(step=step)
        return super().read_index(image_idx, step)

    @dlp.log
    def finalize(self):
        return super().finalize()
    
    def is_index_based(self):
        return True

    def is_iterator_based(self):
        return True
