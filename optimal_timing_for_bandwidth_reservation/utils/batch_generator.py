import torch
from typing import List, Tuple
from config.params import Params

params = Params()


class BatchGenerator:
    """
    A generator that provides batches of input and target data.

    Args:
        data (List[Tuple[torch.Tensor, torch.Tensor]]): The input and target data to batch.
        batch_size (int): The size of each batch.

    Attributes:
        data (List[Tuple[torch.Tensor, torch.Tensor]]): The input and target data.
        batch_size (int): The size of each batch.
        num_batches (int): The total number of batches.

    """

    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size
        self.device = params.DEVICE

    def __len__(self) -> int:
        """
        Returns the total number of batches.
        """
        return self.num_batches

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.idx >= self.num_batches:
            raise StopIteration

        batch_data = self.data[
            self.idx * self.batch_size : (self.idx + 1) * self.batch_size
        ]

        batch_inputs, batch_targets = zip(*batch_data)

        batch_inputs = torch.stack(batch_inputs).to(self.device)
        batch_targets = torch.stack(batch_targets).to(self.device)

        self.idx += 1

        return batch_inputs, batch_targets
