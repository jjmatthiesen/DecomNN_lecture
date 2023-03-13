import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

# ATTENTION: No seed is set .Meaning result vary for each run!

df = pd.read_csv("data/iris.csv")
df.head()

# just for iris data: fake it to binary
df['variety'] = df['variety'].astype('category')
encode_map = {
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 2
}
df['variety'].replace(encode_map, inplace=True)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1


## train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = TrainData(torch.tensor(X_train.values).to(torch.float32),
                       torch.tensor(y_train.values).to(torch.float32))


## test data
class TestData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = TestData(torch.tensor(X_test.values).to(torch.float32),
                     torch.tensor(y_test.values).to(torch.float32))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=10)


class BinaryClassification(nn.Module):
    def __init__(self, inputs_size):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(inputs_size, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

    def extract_features(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        return x


def accuracy(y_pred, y_test):
    y_pred_tag = torch.max(y_pred, 1).indices

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]

    return acc


def get_features(model, loader):
    outputs_features = []
    all_lables = []
    model.eval()
    device = torch.device("cpu")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_features.append(model.extract_features(inputs))
            all_lables.append(labels)
    outputs_features = torch.cat(outputs_features)
    all_lables = torch.cat(all_lables)
    return outputs_features, all_lables


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    num_features = len(X_train.columns)
    model = BinaryClassification(num_features)
    model.to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            y_batch = y_batch.to(int)

            loss = criterion(y_pred, y_batch)
            acc = accuracy(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        print(
            f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    # Todo: save model
    # Todo: put this in a new file when you saved the model
    # Todo: load model

    y_pred_list = []
    acc_list = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_batch = y_batch.to(int)

            acc = accuracy(y_test_pred, y_batch)

            y_pred_tag = torch.max(y_test_pred, 1).indices
            y_pred_list.append(y_pred_tag.cpu().numpy())
            acc_list.append(acc.item())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_list = [item for sublist in y_pred_list for item in sublist]
    confusion_matrix(y_test, y_pred_list)
    acc = np.array(acc_list).mean()
    print("test accuracy is:" + str(acc))

    all_features_test = get_features(model, test_loader)
    
    # perplexity is a hyper paramerter, which should be tuned.
    # For the test_set of the iris data, 10 delivered good results
    tsne = TSNE(n_components=3, perplexity=10).fit_transform(all_features_test[0])
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels_test = all_features_test[1]
    colors = ['#30828A', '#BD5C23', '#84bc23']
    for idx, label in enumerate(labels_test):
        indices = [i for i, l in enumerate(labels_test) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = colors[int(label.item())]
        ax.scatter(current_tx, current_ty, c=color, alpha=0.2, label=label)
    plt.show()

    # alternative plot with df
    # plot tsne
    # df = pd.DataFrame()
    # df["y"] = all_features_test[1]
    # df["comp-1"] = scale_to_01_range(tsne[:, 0])
    # df["comp-2"] = scale_to_01_range(tsne[:, 1])
    #
    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 palette=sns.color_palette("hls", 3),
    #                 data=df).set(title="Iris data T-SNE projection")
    # plt.show()