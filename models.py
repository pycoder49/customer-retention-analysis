import torch


def accuracy(y_pred, y):
    # calculating accuracy
    y = torch.from_numpy(y.values).float().view(-1, 1)
    return (y_pred == y).float().sum() / y_pred.shape[0] * 100


class LinearRegression:
    def __init__(self):
        self.lr = 0.01
        self.w = None
        self.b = None

        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float().view(-1, 1)

        num_features = self.X.shape[1]
        self.w = torch.zeros(num_features, 1)       # shape: [num_features, 1]
        self.b = torch.zeros(1)                     # shape: [1]

        epochs = 300
        losses = []
        for epoch in range(epochs):
            # calculate the linear function
            y_pred = torch.matmul(self.X, self.w) + self.b

            # compute loss
            mse_loss = self._compute_loss()
            losses.append(mse_loss.item())

            # apply gradient descent
            self._linear_grad_descent()

        return losses

    def predict(self, X):
        X = torch.from_numpy(X.values).float()
        return torch.matmul(X, self.w) + self.b

    def predict_class(self, X):
        X = torch.from_numpy(X.values).float()
        y_pred = torch.matmul(X, self.w) + self.b
        return torch.where(y_pred > 0.5, 1, 0)

    def _linear_grad_descent(self):
        # calculate partial derivatives with respect to w and b
        num_samples = self.X.shape[0]
        y_pred = torch.matmul(self.X, self.w) + self.b

        dw = (2 / num_samples) * torch.matmul(self.X.T, y_pred - self.y)
        db = 2 * torch.mean(y_pred - self.y)

        # update the weights
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def _compute_loss(self):
        mse_loss = torch.mean((self.y - (torch.matmul(self.X, self.w) + self.b)) ** 2)
        return mse_loss


class LogisticRegression:
    def __init__(self):
        self.lr = 0.01
        self.lmbda = 0.001
        self.w = None
        self.b = None

        self.X = None
        self.y = None

    def fit(self, X, y):
        # initialize weights, bias, and tensors
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float().view(-1, 1)

        num_features = self.X.shape[1]
        self.w = torch.zeros(num_features, 1)
        self.b = torch.zeros(1)

        epochs = 1000
        losses = []
        for epoch in range(epochs):
            # compute sigmoid activation
            y_pred = (1 + torch.exp(- self._logit())) ** -1

            # compute loss
            log_loss = self._compute_loss()
            losses.append(log_loss.item())

            # apply grad descent
            self._log_grad_descent()

        return losses

    def predict(self, X):
        X = torch.from_numpy(X.values).float()
        logit = torch.matmul(X, self.w) + self.b
        predictions = 1 / (1 + torch.exp(- logit))
        return torch.where(predictions > 0.5, 1, 0)

    def _log_grad_descent(self):
        y_pred = 1 / (1 + torch.exp(- self._logit()))
        num_samples = self.X.shape[0]

        dw = (1/num_samples) * torch.matmul(self.X.T, y_pred - self.y)
        db = torch.mean(y_pred - self.y)

        dw += 0.1 * self.w        # L2 Regularization

        self.w -= self.lr * dw
        self.b -= self.lr * db

    def _compute_loss(self):
        sigmoid = 1 / (1 + torch.exp(- self._logit()))
        first_half = self.y * torch.log(sigmoid + 1e-8)
        second_half = (1 - self.y) * torch.log(1 + sigmoid + 1e-8)
        bce_loss = - torch.mean(first_half + second_half)

        return bce_loss

    def _logit(self):
        return torch.matmul(self.X, self.w) + self.b


class KNearestNeighbor:
    def __init__(self):
        self.k = 3

        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float().view(-1, 1)

    def predict(self, X):
        X = torch.from_numpy(X.values).float()
        y_pred = []
        for i in range(len(X)):
            # get a list of indices of nearest neighbors using euclidean distance
            distance = torch.sqrt(torch.sum((self.X - X[i]) ** 2, dim=1))
            nearest_neighbors = torch.argsort(distance)[:self.k]

            # get the classifications for the nearest indices
            nearest_labels = self.y[nearest_neighbors].squeeze()    # squeeze removes dimensions of 1: [k, 1] -> [k]

            # get the mode of the k classifications
            most_common = torch.mode(nearest_labels).values.item()

            # append it to prediction list
            y_pred.append(most_common)

        return torch.tensor(y_pred).view(-1, 1)


