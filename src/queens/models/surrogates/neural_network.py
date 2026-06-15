#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Feed-forward Neural Network regression model."""

import logging
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from queens.models.surrogates._surrogate import Surrogate
from queens.utils.logger_settings import log_init_args
from queens.utils.random_process_scaler import VALID_SCALER
from queens.utils.valid_options import get_option
from queens.visualization.gaussian_neural_network_vis import plot_loss

_logger = logging.getLogger(__name__)

VALID_ACTIVATIONS = {
    "linear": nn.Identity,
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}

VALID_INITIALIZERS = {
    "glorot_normal": nn.init.xavier_normal_,
    "glorot_uniform": nn.init.xavier_uniform_,
    "he_normal": nn.init.kaiming_normal_,
    "he_uniform": nn.init.kaiming_uniform_,
}


class FeedForwardNet(nn.Module):
    """A general densely connected feed-forward neural network.

    The network can be arbitrary deep and wide and can use different (nonlinear) activation
    functions, optional batch normalization, and optional dropout per hidden layer. The output
    layer is a plain linear layer.

    Attributes:
        network (nn.Sequential): Sequential container holding all layers.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        nodes_per_hidden_layer,
        activation_per_hidden_layer,
        kernel_initializer=None,
        dropout_rate=None,
        batch_norm=False,
    ):
        """Initialize the feed-forward network.

        Args:
            input_dim (int): Dimension of the input layer
            output_dim (int): Dimension of the output layer
            nodes_per_hidden_layer (lst): List containing the number of nodes per hidden layer
            activation_per_hidden_layer (lst): List with strings encoding the activation function
                                               used for the respective hidden layer
            kernel_initializer (str): Type of kernel initialization for the dense layers
            dropout_rate (float): Dropout rate applied after each hidden layer (no dropout if None)
            batch_norm (bool): Whether to add batch normalization after each dense layer
        """
        super().__init__()
        initializer = (
            get_option(VALID_INITIALIZERS, kernel_initializer)
            if kernel_initializer is not None
            else None
        )

        layers = []
        in_features = input_dim
        for num_nodes, activation in zip(nodes_per_hidden_layer, activation_per_hidden_layer):
            num_nodes = int(num_nodes)
            linear = nn.Linear(in_features, num_nodes)
            if initializer is not None:
                initializer(linear.weight)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(num_nodes))
            layers.append(get_option(VALID_ACTIVATIONS, activation)())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            in_features = num_nodes

        output_layer = nn.Linear(in_features, output_dim)
        if initializer is not None:
            initializer(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Network output
        """
        return self.network(x)


class NeuralNetwork(Surrogate):
    """Class for creating a general feed-forward neural network surrogate.

    This is a deterministic, PyTorch-based regression network. It mirrors the structure of the
    :class:`GaussianNeuralNetwork` but replaces the Gaussian output layer with a plain linear
    output layer and uses a mean-squared-error loss. The network can be arbitrary deep and wide
    and model arbitrary nonlinear functions.

    Attributes:
        nn_model (nn.Module): PyTorch based feed-forward neural network model
        num_epochs (int): Number of training epochs
        optimizer_seed (int): Random seed used for initialization of the stochastic gradient
                              descent optimizer
        verbosity_on (bool): Boolean for model verbosity during training. True=verbose
        batch_size (int): Size of data-batch (smaller than the training data size)
        scaler_x (obj): Scaler for inputs
        scaler_y (obj): Scaler for outputs
        loss_plot_path (str): Path to determine whether loss plot should be produced
                              (yes if provided). Plot will be saved at path location.
        num_refinements (int): Number of refinements
        refinement_epochs_decay (float): Decrease of epochs in refinements
        mean_function (function): Mean function of the Neural Network
        gradient_mean_function (function): Gradient of the mean function of the Neural Network
        adams_training_rate (float): Training rate for the ADAMS gradient descent optimizer
        nodes_per_hidden_layer (lst): List containing number of nodes per hidden layer of the
                                      Neural Network. The length of the list defines the deepness
                                      of the model and the values the width of the individual
                                      layers.
        activation_per_hidden_layer (list): List with strings encoding the activation function that
                                            shall be used for the respective hidden layer of the
                                            Neural Network
        kernel_initializer (str): Type of kernel initialization for the neural network
        dropout_rate (float): Dropout rate applied after each hidden layer (no dropout if None)
        batch_norm (bool): Whether to add batch normalization after each dense layer
        num_validation_data (int): Number of validation samples taken from training data
    """

    @log_init_args
    def __init__(
        self,
        num_epochs=None,
        batch_size=None,
        adams_training_rate=None,
        optimizer_seed=None,
        verbosity_on=None,
        nodes_per_hidden_layer_lst=None,
        activation_per_hidden_layer_lst=None,
        kernel_initializer=None,
        loss_plot_path=False,
        refinement_epochs_decay=0.75,
        data_scaling=None,
        mean_function_type="zero",
        dropout_rate=None,
        batch_norm=False,
        num_validation_data=None,
        training_iterator=None,
    ):
        """Initialize an instance of the feed-forward Neural Network.

        Args:
            num_epochs (int): Number of epochs used for training of the network
            batch_size (int): Size of data-batch (smaller than the training data size)
            adams_training_rate (float): Training rate for the ADAMS gradient descent optimizer
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose
            nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of
                                          the Neural Network. The length of the list defines the
                                          deepness of the model and the values the width of the
                                          individual layers.
            activation_per_hidden_layer_lst (list): List with strings encoding the activation
                                                function that shall be used for the respective
                                                hidden layer of the Neural Network
            kernel_initializer (str): Type of kernel initialization for the neural network
            loss_plot_path (str): Path to determine whether loss plot should be produced
                                  (yes if provided). Plot will be saved at path location.
            refinement_epochs_decay (float): Decrease of epochs in refinements
            data_scaling (str): Data scaling type
            mean_function_type (str): Mean function type of the Neural Network
            dropout_rate (float): Dropout rate applied after each hidden layer (no dropout if None)
            batch_norm (bool): Whether to add batch normalization after each dense layer
            num_validation_data (int): Number of validation samples taken from training data
            training_iterator (Iterator): Iterator for training data
        Returns:
            Instance of NeuralNetwork
        """
        super().__init__(training_iterator=training_iterator)
        # check mean function and subtract from y_train
        self.valid_mean_function_types = {
            "zero": (lambda x: 0, lambda x: 0),
            "identity_multi_fidelity": (
                lambda x: np.atleast_2d(x[:, 0]).T,
                lambda x: np.hstack((np.ones(x.shape[0]).reshape(-1, 1), np.zeros(x[:, 1:].shape))),
            ),
        }

        self.mean_function_type = mean_function_type
        self.mean_function = None
        self.gradient_mean_function = None
        self.nn_model = None
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.batch_size = batch_size
        self.data_scaling = data_scaling
        self.scaler_x = None
        self.scaler_y = None
        self.loss_plot_path = loss_plot_path
        self.num_refinements = 0
        self.refinement_epochs_decay = refinement_epochs_decay

        self.adams_training_rate = adams_training_rate
        self.nodes_per_hidden_layer = nodes_per_hidden_layer_lst
        self.activation_per_hidden_layer = activation_per_hidden_layer_lst
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.num_validation_data = num_validation_data
        self.validation_data = None

    def _build_model(self):
        """Build the feed-forward neural network.

        We use a regular densely connected NN with a plain linear output layer. The network can be
        arbitrary deep and wide and can use different (nonlinear) activation functions.

        Returns:
            model (FeedForwardNet): PyTorch model instance
        """
        input_dim = self.x_train.shape[1]
        output_dim = self.y_train.shape[1]

        model = FeedForwardNet(
            input_dim=input_dim,
            output_dim=output_dim,
            nodes_per_hidden_layer=self.nodes_per_hidden_layer,
            activation_per_hidden_layer=self.activation_per_hidden_layer,
            kernel_initializer=self.kernel_initializer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
        )

        return model

    def update_training_data(self, x_train, y_train):
        """Update the training data of the model.

        Args:
            x_train (np.array): Training input array
            y_train (np.array): Training output array
        """
        num_old_samples = self.x_train.shape[0]
        x_train_new = self.scaler_x.transform(x_train[num_old_samples:].T).T
        y_train_new = self.scaler_y.transform(y_train[num_old_samples:])
        self.x_train = np.vstack((self.x_train, x_train_new))
        self.y_train = np.vstack((self.y_train, y_train_new))

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        self.mean_function, self.gradient_mean_function = get_option(
            self.valid_mean_function_types, self.mean_function_type, "mean_function_type"
        )
        self.scaler_x = get_option(VALID_SCALER, self.data_scaling)()
        self.scaler_y = get_option(VALID_SCALER, self.data_scaling)()

        y_train = y_train - self.mean_function(x_train)

        self.scaler_x.fit(x_train)
        self.x_train = self.scaler_x.transform(x_train)
        self.scaler_y.fit(y_train)
        self.y_train = self.scaler_y.transform(y_train)

        self.validation_data = None
        if self.num_validation_data is not None:
            if self.num_validation_data <= 0:
                raise ValueError("num_validation_data must be positive when provided.")
            if self.num_validation_data >= self.x_train.shape[0]:
                raise ValueError("num_validation_data must be smaller than training set size.")
            self.x_train, x_val, self.y_train, y_val = train_test_split(
                self.x_train,
                self.y_train,
                test_size=self.num_validation_data,
                random_state=42,
                shuffle=True,
            )
            self.validation_data = (x_val, y_val)

        self.nn_model = self._build_model()

    def train(self):
        """Train the feed-forward neural network.

        The network is trained with the Adamax optimizer minimizing the mean-squared-error loss.
        Early stopping based on the validation loss restores the best weights when the loss starts
        to increase again over several iterations.
        """
        # make epochs adaptive with a simple schedule, lower bound is 1/5 of the initial epoch
        if self.num_refinements > 0:
            self.num_epochs = int(
                max(self.num_epochs * self.refinement_epochs_decay, self.num_epochs / 5)
            )
        self.num_refinements += 1

        # set the random seeds for optimization/training
        torch.manual_seed(self.optimizer_seed)
        np.random.seed(self.optimizer_seed)

        x_train = torch.as_tensor(self.x_train, dtype=torch.float32)
        y_train = torch.as_tensor(self.y_train, dtype=torch.float32)
        dataset = TensorDataset(x_train, y_train)
        batch_size = self.batch_size if self.batch_size is not None else len(dataset)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        validation_data = None
        if self.validation_data is not None:
            x_val, y_val = self.validation_data
            validation_data = (
                torch.as_tensor(x_val, dtype=torch.float32),
                torch.as_tensor(y_val, dtype=torch.float32),
            )

        optimizer = torch.optim.Adamax(self.nn_model.parameters(), lr=self.adams_training_rate)
        loss_function = nn.MSELoss()

        patience = 1000
        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.num_epochs):
            self.nn_model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()
                prediction = self.nn_model(x_batch)
                loss = loss_function(prediction, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1.0e3)
                optimizer.step()
                epoch_loss += loss.item() * x_batch.shape[0]
            epoch_loss /= len(dataset)
            train_loss_history.append(epoch_loss)

            # monitor loss for early stopping (validation loss if available, else training loss)
            monitor_loss = epoch_loss
            if validation_data is not None:
                self.nn_model.eval()
                with torch.no_grad():
                    val_prediction = self.nn_model(validation_data[0])
                    monitor_loss = loss_function(val_prediction, validation_data[1]).item()
                val_loss_history.append(monitor_loss)

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = {k: v.detach().clone() for k, v in self.nn_model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.verbosity_on:
                _logger.info("Epoch %d/%d - loss: %.6e", epoch + 1, self.num_epochs, monitor_loss)

            if epochs_without_improvement >= patience:
                _logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        # restore best weights
        if best_state is not None:
            self.nn_model.load_state_dict(best_state)

        _logger.info(self.nn_model)

        if self.loss_plot_path:
            history_dict = {"loss": train_loss_history}
            if val_loss_history:
                history_dict["val_loss"] = val_loss_history
            plot_loss(SimpleNamespace(history=history_dict), self.loss_plot_path)

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t.

        current set of input samples.
        """
        raise NotImplementedError

    def predict(self, x_test, support="y", gradient_bool=False):
        """Predict the output at x_test.

        Args:
            x_test (np.array): Testing input vector for which the point estimates should be
                               predicted
            support (str, optional): String to define the support of the output
                                    - 'y': Output is defined on the output space
                                    - 'f': Output is defined on the latent space
            gradient_bool (bool, optional): Boolean to configure whether gradients should be
                                            returned as well

        Returns:
            output (dict): Dictionary with output statistics
        """
        if support == "f":
            raise NotImplementedError('Support "f" is not implemented yet.')

        if gradient_bool:
            output = self.predict_and_gradient(x_test)
        else:
            output = self.predict_y(x_test)

        output["x_test"] = x_test
        return output

    def predict_y(self, x_test):
        """Predict the output mean.

        Prediction is conducted w.r.t. to the output space "y".

        Args:
            x_test (np.array): Testing input vector for which the point estimates should be
                               predicted

        Returns:
            output (dict): Dictionary with output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        x_test_tensor = torch.as_tensor(x_test_transformed, dtype=torch.float32)
        self.nn_model.eval()
        with torch.no_grad():
            yhat = self.nn_model(x_test_tensor).numpy()

        output = {
            "result": self.scaler_y.inverse_transform_mean(yhat).reshape(-1, 1)
            + self.mean_function(x_test)
        }

        return output

    def predict_and_gradient(self, x_test):
        """Predict the mean and its gradient at x_test.

        Args:
            x_test (np.array): Testing input vector for which the point estimates should be
                               predicted

        Returns:
            output (dict): Dictionary with output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        x_test_tensor = torch.as_tensor(
            x_test_transformed, dtype=torch.float32, device=None
        ).requires_grad_(True)

        self.nn_model.eval()
        mean_pred = self.nn_model(x_test_tensor)
        grad_mean = torch.autograd.grad(mean_pred.sum(), x_test_tensor)[0].numpy()
        mean_pred = mean_pred.detach().numpy()

        output = {
            "result": self.scaler_y.inverse_transform_mean(mean_pred).reshape(-1, 1)
            + self.mean_function(x_test)
        }
        output["grad_mean"] = self.scaler_y.inverse_transform_grad_mean(
            grad_mean, self.scaler_x.standard_deviation
        ) + self.gradient_mean_function(x_test)

        return output

    def save(self, path):
        """Save the model to a specified path.

        Args:
            path (str): Path where the model should be saved
        """
        path.mkdir(exist_ok=True)

        torch.save(self.nn_model.state_dict(), path / "nn_model.pt")
        self.scaler_x.save(path / "scaler_x.npz")
        self.scaler_y.save(path / "scaler_y.npz")
        np.savez(
            path / "metadata.npz",
            mean_function_type=self.mean_function_type,
            data_scaling=self.data_scaling,
            input_dim=self.x_train.shape[1],
            output_dim=self.y_train.shape[1],
            nodes_per_hidden_layer=np.array(self.nodes_per_hidden_layer),
            activation_per_hidden_layer=np.array(self.activation_per_hidden_layer),
            kernel_initializer=str(self.kernel_initializer),
            dropout_rate=np.nan if self.dropout_rate is None else self.dropout_rate,
            batch_norm=self.batch_norm,
        )

    def load(self, path):
        """Load the model from a specified path.

        Args:
            path (str): Path where the model should be loaded from
        """
        metadata = np.load(path / "metadata.npz")
        self.mean_function_type = str(metadata["mean_function_type"])
        self.data_scaling = str(metadata["data_scaling"])
        self.nodes_per_hidden_layer = list(metadata["nodes_per_hidden_layer"])
        self.activation_per_hidden_layer = [
            str(activation) for activation in metadata["activation_per_hidden_layer"]
        ]
        kernel_initializer = str(metadata["kernel_initializer"])
        self.kernel_initializer = None if kernel_initializer == "None" else kernel_initializer
        dropout_rate = float(metadata["dropout_rate"])
        self.dropout_rate = None if np.isnan(dropout_rate) else dropout_rate
        self.batch_norm = bool(metadata["batch_norm"])

        self.mean_function, self.gradient_mean_function = get_option(
            self.valid_mean_function_types, self.mean_function_type, "mean_function_type"
        )

        self.nn_model = FeedForwardNet(
            input_dim=int(metadata["input_dim"]),
            output_dim=int(metadata["output_dim"]),
            nodes_per_hidden_layer=self.nodes_per_hidden_layer,
            activation_per_hidden_layer=self.activation_per_hidden_layer,
            kernel_initializer=self.kernel_initializer,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
        )
        self.nn_model.load_state_dict(torch.load(path / "nn_model.pt"))
        self.nn_model.eval()

        self.scaler_x = get_option(VALID_SCALER, self.data_scaling)()
        self.scaler_y = get_option(VALID_SCALER, self.data_scaling)()
        self.scaler_x.load(path / "scaler_x.npz")
        self.scaler_y.load(path / "scaler_y.npz")

        self.is_trained = True
