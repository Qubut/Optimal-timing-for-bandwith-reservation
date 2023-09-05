"""
This module provides a BatchGenerator class that generates batches of input and target data.

Classes:
- BatchGenerator

Dependencies:
- torch
- typing.List
- typing.Tuple

"""


import torch
from typing import List, Tuple


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_batches = len(data) // batch_size

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of batches.
        """
        return self.num_batches

    def __iter__(self):
        """
        Returns:
            BatchGenerator: The BatchGenerator object.
        """
        # initialize the batch index
        self.idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of batched input and target data.
        """

        # check if we have reached the end of the data
        if self.idx >= self.num_batches:
            # stop iteration
            raise StopIteration

        # select the batch
        batch_data = self.data[
            self.idx * self.batch_size : (self.idx + 1) * self.batch_size
        ]

        # get the input data and add an extra dimension at the end
        batch_input = [d[0] for d in batch_data]


        # get the target data
        batch_target = [d[1] for d in batch_data]

        # stack the input data along the last dimension and move it to the device
        batch_input = torch.cat(batch_input, dim=1).to(self.device)


        # stack the target data and flatten it, then move it to the device
        batch_target = torch.stack(batch_target).flatten().to(self.device)

        # increment the batch index
        self.idx += 1
        return batch_input, batch_target
