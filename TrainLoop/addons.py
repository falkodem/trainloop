import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ Orthogonal Regularization Example --------------------------
class OrthogonalRegularization:
    def __init__(self, lambda_ortho=1e-3):
        self.lambda_ortho = lambda_ortho

    def __call__(self, model):
        ortho_loss = 0.0
        for param in model.parameters():
            if param.dim() == 2:  # For fully connected layers
                # Compute the Gram matrix
                gram_matrix = torch.mm(param.t(), param)
                # Off-diagonal elements
                off_diagonal = gram_matrix - torch.diag(gram_matrix.diagonal())
                # Orthogonality loss
                ortho_loss += torch.sum(off_diagonal ** 2)
            elif param.dim() == 4:  # For convolutional layers
                # Reshape to (out_channels, in_channels  kernel_height  kernel_width)
                n_out, n_in, kh, kw = param.shape
                reshaped_param = param.view(n_out, -1)  # Shape: (out_channels, in_channels  kernel_height  kernel_width)
                
                # Compute the Gram matrix
                gram_matrix = torch.mm(reshaped_param.t(), reshaped_param)
                # Off-diagonal elements
                off_diagonal = gram_matrix - torch.diag(gram_matrix.diagonal())
                # Orthogonality loss
                ortho_loss += torch.sum(off_diagonal ** 2)

        return self.lambda_ortho * ortho_loss

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32  7  7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
ortho_reg = OrthogonalRegularization(lambda_ortho=1e-3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add orthogonal regularization loss
        ortho_loss = ortho_reg(model)
        total_loss = loss + ortho_loss
        
        total_loss.backward()
        optimizer.step()
