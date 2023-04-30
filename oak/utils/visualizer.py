import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.features = dict()
        self.layers = layers

        for layer in layers:
            model.get_submodule(layer).register_forward_hook(self.get_features(layer))

    def forward(self, x):
        x = self.model(x)
        return x

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook

    def collect_features(self, dm):
        features = dict()
        for idx, inputs in enumerate(dm):
            x, y = inputs
            output = self.model(x)

            B, C, H, W = x.shape

            if idx == 0:
                features['input'] = x.view(B, -1)
                features['labels'] = y.view(B, 1)
                features['output'] = output.detach().cpu()
                for layer in self.layers:
                    features[layer] = self.features[layer].cpu()

            else:
                features['input'] = torch.concatenate((features['input'], x.view(B, -1)), dim=0)
                features['labels'] = torch.concatenate((features['labels'], y.view(-1, 1)), dim=0)
                features['output'] = torch.concatenate((features['output'], output.detach().cpu()), dim=0)
                for layer in self.layers:
                    features[layer] = torch.concatenate((features[layer], self.features[layer].cpu()), dim=0)

        return features

    def PCA(self, features, layers=None, k=3):
        if layers is None:
            layers = features.keys()

        PCA = dict()
        for layer in layers:
            U, S, V = torch.pca_lowrank(features[layer], q=k)
            projection = torch.matmul(features[layer], V[:, :k])
            approx = torch.matmul(projection, V.T)

            PCA[layer] = dict()
            PCA[layer]['proj'] = projection
            PCA[layer]['acc_var'] = torch.var(approx) / torch.var(projection) * 100

        return PCA

    def PCA_scatter(self, features, layers=None, k=3):
        assert k==2 or k==3, 'k must be 2 or 3'
        PCA = self.PCA(features, layers=layers, k=k)
        for layer in layers:
            num_classes = len(torch.unique(features['labels']))
            f = PCA[layer]['proj']
            if k==3:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig, ax = plt.subplots()
            sm = ax.scatter(f[:, 0], f[:, 1], f[:, 2], c=features['labels'], cmap='Spectral')
            plt.title(f'{layer} PCA')
            plt.colorbar(sm, boundaries=np.arange(num_classes+1) - 0.5).set_ticks(np.arange(num_classes))
        plt.show()
