# PyTorch Model Templates for Quick Conversion

# Model 2: CNN with Augmentation (same architecture as Model 1, different input size)
class CNNWithAug(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(CNNWithAug, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.20)

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # For 128x128 input: 128 -> 64 -> 32 -> 16
        self.flatten_size = 64 * 16 * 16

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout2(x)

        x = x.view(-1, self.flatten_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Model 3: CNN-LSTM
class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, lstm_hidden=100, num_classes=2):
        super(CNNLSTM, self).__init__()

        # CNN layers (TimeDistributed)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 128x128 -> 64x64 -> 32x32
        self.flatten_size = 32 * 32 * 32

        # LSTM layer
        self.lstm = nn.LSTM(self.flatten_size, lstm_hidden, batch_first=True)

        # Output layer
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x shape: (batch, timesteps, channels, height, width)
        batch_size, timesteps, C, H, W = x.size()

        # Process each timestep through CNN
        c_out = []
        for t in range(timesteps):
            # CNN for this timestep
            c = F.relu(self.conv1(x[:, t, :, :, :]))
            c = self.pool1(c)
            c = F.relu(self.conv2(c))
            c = self.pool2(c)
            c = c.view(batch_size, -1)  # Flatten
            c_out.append(c)

        # Stack timesteps
        lstm_input = torch.stack(c_out, dim=1)  # (batch, timesteps, features)

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Output
        out = self.fc(last_output)
        return out


# Model 4: CNN-SVM (CNN with L2 regularization)
class CNNSVM(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(CNNSVM, self).__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 224x224 -> 112x112 -> 56x56
        self.flatten_size = 32 * 56 * 56

        # SVM-like classifier (Dense with L2 regularization)
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return x
