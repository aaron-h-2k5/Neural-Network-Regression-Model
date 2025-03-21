# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This project aims to develop a Neural Network Regression Model capable of accurately predicting a target variable based on input features. By utilizing deep learning techniques, the model will learn complex patterns within the dataset to deliver reliable predictions.

## Neural Network Model
<img width="764" alt="Screenshot 2025-03-21 at 8 57 28 AM" src="https://github.com/user-attachments/assets/3ac9f64c-35d8-4b02-a22c-f7cc569ff9cd" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Aaron H
### Register Number: 212223040001
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information
<img width="76" alt="Screenshot 2025-03-21 at 8 55 11 AM" src="https://github.com/user-attachments/assets/a75f2637-69fa-4480-8c04-bf9c1a198696" />

## OUTPUT

### Training Loss Vs Iteration Plot
<img width="593" alt="Screenshot 2025-03-21 at 8 55 42 AM" src="https://github.com/user-attachments/assets/4d4c0017-b586-45a6-9125-e0a7b0befe13" />

### New Sample Data Prediction
<img width="858" alt="Screenshot 2025-03-21 at 8 56 04 AM" src="https://github.com/user-attachments/assets/1c7bbd76-4b3b-470e-a0db-0dc9cb74b87b" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
