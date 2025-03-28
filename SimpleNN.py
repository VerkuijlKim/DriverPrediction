import pandas as pd
import torch
from torch import optim
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from DataDefined2 import *
from Functions2 import *

CLASS_OOD = 1
THRESHOLD = 0.65
EPOCHS = 500


class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, 10)   # Output layer: 10 classes
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data():
    df = pd.read_csv('Driving Data(KIA SOUL)_(150728-160714)_(10 Drivers_A-J).csv')
    # Dropping columns
    columns_to_drop = one_val_col + irrelevant_col
    df = df.drop(columns_to_drop, axis=1)
    # Adding columns
    df = addRideNumbers(df)
    df = add_delta(df, delta_col)

    # Print rides per class
    for driver_class in sorted(df['Class'].unique()):
        driver_data = df[df['Class'] == driver_class]
        ride_counts = driver_data['Ride number'].unique()
        print(f"Ride number for Class {driver_class}: {ride_counts}")
    
    encoder = LabelEncoder()
    df['Class'] = encoder.fit_transform(df['Class'])

    train_df, test_df = split_train_test_ood(df, CLASS_OOD, frac=0.15)

    # Normalize
    columns_to_scale = numerical_col + two_val_col
    train_df, test_df = normalize_data(train_df, test_df, columns_to_scale)
    # Encode
    train_df, test_df = label_encode(train_df, test_df, categorical_col)

    print("Classes in train:", len(train_df['Class'].unique()))
    print("Classes in test:", len(test_df['Class'].unique()))

    # Convert to input for the model
    X_train = torch.tensor(train_df[numerical_col + ['Indication_of_brake_switch_ON/OFF']].values, dtype=torch.float32)
    X_test = torch.tensor(test_df[numerical_col + ['Indication_of_brake_switch_ON/OFF']].values, dtype=torch.float32)
    y_train = torch.tensor(train_df['Class'].values, dtype=torch.long)
    y_test = torch.tensor(test_df['Class'].values, dtype=torch.long)

    return X_train, X_test, y_train, y_test


def evaluation(y_test, y_pred):
    # To evaluate, the OOD prediction should be compared
    # Therefore, change class number that is OOD in the testset to -1
    y_test_ood = torch.where(y_test == CLASS_OOD, -1, y_test)

    print(classification_report(y_test_ood, y_pred))

    conf_matrix = confusion_matrix(y_test_ood, y_pred)
    plt.figure(figsize=(10,7))
    # (Hardcoded, only works when CLASS_OOD = 1)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels= ['-1 (OOD)',  'A',  'C',  'D',  'E',  'F',  'G',  'H',  'I',  'J'], 
                yticklabels= ['B',  'A',  'C',  'D',  'E',  'F',  'G',  'H',  'I',  'J'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()



if __name__== "__main__":
    X_train, X_test, y_train, y_test = load_data()

    input_dim = X_train.shape[1]   
    model = TabularNN(input_dim)
    criterion = nn.CrossEntropyLoss()  # For multi-class classifications
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train)  # Logits output
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Predict on test data
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)  # Logits
        # Convert logits to probabilities (using softmax) and get predicted class
        predicted_probs = torch.softmax(test_predictions, dim=1)
        max_probs, predicted_classes = torch.max(predicted_probs, dim=1)
        predicted_classes[max_probs < THRESHOLD] = -1

    evaluation(y_test, predicted_classes)