"""
This module contains the Trainer class for training PyTorch models.

Classes:
Trainer: A class for training PyTorch models.

Dependencies:
- torch
"""

import torch


class Trainer:
    """
    A class for training PyTorch models.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_fn (callable): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used during training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): An optional learning rate scheduler.

    Methods:
        train(data_loader): Trains the model on the provided data.
        train_with_scheduler(data_loader): Trains the model on the provided data with learning rate scheduling.
    """

    def __init__(self, model, loss_fn, optimizer, device, scheduler=None):
        """
        Constructs a `Trainer` instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            loss_fn (callable): The loss function to be used during training.
            optimizer (torch.optim.Optimizer): The optimizer to be used during training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): An optional learning rate scheduler.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def _train_iteration(self, seq, labels):
        """
        Performs a single training iteration.

        Args:
            seq (torch.Tensor): The input sequence tensor.
            labels (torch.Tensor): The target label tensor.

        Returns:
            The loss of the current iteration.
        """
        seq, labels = seq.to(self.device), labels.to(self.device)
        y_pred = self.model(seq).squeeze()
        loss = self.loss_fn(y_pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
        self.optimizer.step()
        return loss.item()

    def train(self, data_loader):
        """
        Trains the model on the provided data.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the training data.

        Returns:
            The average training loss per batch.
        """
        tot_loss = 0.0
        for seq, labels in data_loader:
            tot_loss += self._train_iteration(seq, labels)
        return tot_loss / len(data_loader)

    def train_with_scheduler(self, data_loader):
        """
        Trains the model on the provided data with learning rate scheduling.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the training data.

        Returns:
            The average training loss per batch.
        """
        tot_loss = 0.0
        for seq, labels in data_loader:
            tot_loss += self._train_iteration(seq, labels)
        if self.scheduler is not None:
            self.scheduler.step()
        return tot_loss / len(data_loader)
