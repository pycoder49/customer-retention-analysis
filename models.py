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
    pass


class KNearestNeighbor:
    pass

