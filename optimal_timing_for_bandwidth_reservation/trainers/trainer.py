import numpy as np
import torch

class Trainer:
    """
    A class for training PyTorch models.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained.
        loss_fn (callable): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used during training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): An optional learning rate scheduler.
        device: The device type on which the model is trained.
        is_transformer: Boolean indicating if the model is a transformer.

    Methods:
        train(data_loader): Trains the model on the provided data.
        train_with_scheduler(data_loader): Trains the model on the provided data with learning rate scheduling.
    """

    def __init__(
        self, model, loss_fn, optimizer, device, scheduler=None, is_transformer=False
    ):
        """
        Constructs a `Trainer` instance.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            loss_fn (callable): The loss function to be used during training.
            optimizer (torch.optim.Optimizer): The optimizer to be used during training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): An optional learning rate scheduler.
            device: The device type on which the model is trained.
            is_transformer: Boolean indicating if the model is a transformer.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.is_transformer = is_transformer

    def compute_accuracy(self, outputs, labels):
        """
        Computes the accuracy based on model's outputs and actual labels.

        Args:
            outputs (torch.Tensor): Model's output.
            labels (torch.Tensor): True labels.

        Returns:
            float: Computed accuracy.
        """
        predicted = outputs.round()  # For simplicity, round to nearest integer
        correct = (predicted == labels).float().sum()
        accuracy = correct / len(labels)
        return accuracy.item()

    def _train_iteration(self, seq, labels):
        """
        Performs a single training iteration.

        Args:
            seq (torch.Tensor): The input sequence tensor.
            labels (torch.Tensor): The target label tensor.

        Returns:
            The loss and accuracy of the current iteration.
        """
        seq, labels = seq.to(self.device), labels.to(self.device)
        
        if self.is_transformer:
            src_mask = self.model.generate_square_subsequent_mask(seq.size(0))
            y_pred = self.model(seq, src_mask).squeeze()
        else:
            y_pred = self.model(seq).squeeze()
            
        loss = self.loss_fn(y_pred, labels)
        accuracy = self.compute_accuracy(y_pred, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
        self.optimizer.step()
        
        return loss.item(), accuracy

    def train(self, data_loader):
        """
        Trains the model on the provided data.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the training data.

        Returns:
            The average training loss and accuracy per batch.
        """
        tot_loss = 0.0
        tot_accuracy = 0.0
        length =  len(data_loader)
        for seq, labels in data_loader:
            loss, accuracy = self._train_iteration(seq, labels)
            tot_loss += loss
            tot_accuracy += accuracy
            
        avg_loss = tot_loss / length
        avg_accuracy = tot_accuracy / length
        return avg_loss, avg_accuracy

    def train_with_scheduler(self, data_loader):
        """
        Trains the model on the provided data with learning rate scheduling.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the training data.

        Returns:
            The average training loss and accuracy per batch.
        """
        tot_loss = 0.0
        tot_accuracy = 0.0
        
        for seq, labels in data_loader:
            loss, accuracy = self._train_iteration(seq, labels)
            tot_loss += loss
            tot_accuracy += accuracy
            
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = tot_loss / len(data_loader)
        avg_accuracy = tot_accuracy / len(data_loader)
        return avg_loss, avg_accuracy
