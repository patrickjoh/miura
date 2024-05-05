import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler as Scaler


class LSTM_inc(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1, num_epochs=50, learning_rate=0.005):
        super(LSTM_inc, self).__init__()

        # Setting random seed for reproducibility
        torch.manual_seed(140)
        np.random.seed(140)
        random.seed(140)
        
        # Initialize scalers
        self.sc_x = Scaler()
        self.sc_y = Scaler()
        
        # LSTM Model Configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Define LSTM and FC layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def fit_transform(self, x_values, y_values):
        # Scale the data
        x_values_2D = x_values.squeeze(-1).numpy()
        y_values_2D = y_values.squeeze(-1).numpy()
        
        x_scaled = self.sc_x.fit_transform(x_values_2D)
        y_scaled = self.sc_y.fit_transform(y_values_2D)
        
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(-1)
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(-1)
        
        return x_scaled, y_scaled
    
    def transform(self, x_values, y_values):
        x_values_2D = x_values.squeeze(-1).numpy()
        y_values_2D = y_values.squeeze(-1).numpy()
        
        x_scaled = self.sc_x.transform(x_values_2D)
        y_scaled = self.sc_y.transform(y_values_2D)
        
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(-1)
        y_scaled = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(-1)
        
        return x_scaled, y_scaled
    
    def transform_x(self, x_values):
        x_values_2D = x_values.squeeze(-1).numpy()
        x_scaled = self.sc_x.transform(x_values_2D)
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(-1)
        return x_scaled

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM output
        x = self.fc(x[:, -1, :])  # Extract only the last timestep's output for prediction
        return x

    def inverse_transform_y(self, y_pred):
        # Convert predictions back to original scale
        y_pred_np = y_pred.detach().numpy()  # Convert to numpy array
        y_pred_orig = self.sc_y.inverse_transform(y_pred_np)
        return torch.tensor(y_pred_orig, dtype=torch.float32)

    def train_model(self, x, y):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss

        for epoch in range(self.num_epochs):
            self.train()  # Set the model to training mode
            output = self(x)  # Forward pass
            optimizer.zero_grad()  # Clear the gradients
            loss = loss_fn(output, y.view(-1, 1))  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

        #    if epoch % 5 == 0:  # Print the loss every 5 epochs
        #        print(f'Epoch [{epoch}/{self.num_epochs}], Loss: {loss.item()}')


    ######################## Public methods ########################
                
    def initialize(self, x_values, y_values):
        # Initialize scalers
        x_scaled, y_scaled = self.fit_transform(x_values, y_values)
        # Train the model
        self.train_model(x_scaled, y_scaled)

        # Return the trained model
        return self

    def continue_training(self, x_values, y_values):
        # Scale the data
        x_scaled, y_scaled = self.transform(x_values, y_values)
        # Train the model
        self.train_model(x_scaled, y_scaled)

        # Return the trained model
        return self
    
    def predict_next(self):
         # Get a prediction
        x = torch.tensor([4], dtype=torch.float32).view(-1, 1, 1)
        x = self.transform_x(x)
        y_pred = self(x)
        y_pred = self.inverse_transform_y(y_pred)
        return y_pred.item()