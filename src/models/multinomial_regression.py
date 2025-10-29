import numpy as np
from tqdm import tqdm

from pathlib import Path
import matplotlib.pyplot as plt


class DataProvider:
    def __init__(self, X:np.ndarray, y:np.ndarray, batch_size=100, shuffle=True) -> None:
        assert X.shape[0] == y.shape[0]
        assert batch_size < X.shape[0]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0

        self.batches = self.build_batches(self.X, self.y, self.batch_size)
        pass
    
    def build_batches(self, X, y, batch_size):
        if self.shuffle == True:
            idx_arr = np.arange(X.shape[0])
            np.random.shuffle(idx_arr)
            X = X[idx_arr]
            y = y[idx_arr]
        batches = []
        for i in range(X.shape[0]//batch_size):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]
            batch = (X_batch, y_batch)
            batches.append(batch)
        return batches

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self.batch = self.build_batches(self.X, self.y, self.batch_size)
        return self
    
    def __next__(self):
        if self.index >= len(self.batches):
            raise StopIteration
        
        batch = self.batches[self.index]
        self.index += 1
        return batch
    
    def __len__(self):
        return len(self.batches)


class MultinomialRegressor:
    def __init__(self, n_clases:int, n_inputs:int) -> None:
        self.weights = np.random.normal(loc=0, scale=0.01, size=(n_inputs, n_clases))
        self.bias = np.random.normal(loc=0, scale=0.01, size=n_clases)

    def softmax(self, y):
        if y.ndim == 1:
            y = y.reshape(1, -1)
        exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def forward(self, X:np.ndarray):
        return self.softmax(X.dot(self.weights) + self.bias)

    def train(self, train_data:DataProvider, learning_rate=1e-3, num_epochs=50):
        assert isinstance(train_data, DataProvider)

        num_batches = len(train_data)
        bar = tqdm(total=num_epochs)
        loss = []
        for epoch in range(num_epochs):
            bar.update(n=1)
            epoch_loss = 0
            for X, y in train_data:
                y_hat = self.forward(X) 
                N = y.shape[0]

                eps = 1e-9
                epoch_loss += -np.mean(np.sum(y * np.log(y_hat + eps), axis=1))

                grad_w = X.T.dot(y_hat - y) / N
                grad_b = np.mean(y_hat - y, axis=0)

                self.weights -= learning_rate * grad_w
                self.bias -= learning_rate * grad_b
            loss.append(epoch_loss/num_batches)
        return loss


def train_test_split(X, y, test_ratio=0.2):
    """
    Splits arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels of shape (n_samples,) or (n_samples, n_outputs).
    test_ratio : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    assert len(X) == len(y), "X and y must have the same length"
    n_samples = len(X)
    test_size = int(n_samples * test_ratio)

    # Generate shuffled indices
    indices = np.arange(n_samples)

    # Split into train/test
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)


ROOT = Path(__file__).resolve().parent.parent.parent
data = np.load(ROOT / 'data/species_train.npz')

num_class = 500
num_inputs = 2

id_idx_lookup = {id:idx for idx,id in enumerate(data['taxon_ids'])}
X_data = data['train_locs']
y_data = np.array([id_idx_lookup[id] for id in data['train_ids']])
y_data_on_hot = np.eye(num_class)[y_data]

train_data, test_data = train_test_split(X_data, y_data_on_hot, test_ratio=0.1)
data_provider = DataProvider(train_data[0], train_data[1])

model = MultinomialRegressor(num_class, num_inputs)
loss = model.train(data_provider, num_epochs=100)

plt.plot(np.arange(len(loss)), loss)
plt.show()

random_idx = np.random.randint(low=0, high=test_data[1].shape[0])
test_point = (test_data[0][random_idx], test_data[1][random_idx])

probs = np.squeeze(model.forward(test_point[0]))
true_idx = np.argmax(test_point[1])


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: Model predictions ---
axes[0].bar(np.arange(len(probs)), probs, color="skyblue")
axes[0].set_title("Predicted probability distribution")
axes[0].set_xlabel("Class index")
axes[0].set_ylabel("Predicted probability")

# --- Right: True label (one-hot or index) ---
true_onehot = np.zeros_like(probs)
true_onehot[true_idx] = 1.0

axes[1].bar(np.arange(len(true_onehot)), true_onehot, color="salmon")
axes[1].set_title(f"True label distribution (True index = {true_idx})")
axes[1].set_xlabel("Class index")
axes[1].set_ylabel("True probability")

plt.tight_layout()
plt.show()