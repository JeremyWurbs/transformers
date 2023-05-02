import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio


class Visualizer(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self._features = dict()
        self.layers = layers

        for layer in layers:
            model.get_submodule(layer).register_forward_hook(self.get_features(layer))

    def forward(self, x):
        x = self.model(x)
        return x

    def get_features(self, name):
        def hook(model, input, output):
            self._features[name] = output.detach().cpu()
            if name.find('mhsa.head') >= 0:
                model.forward(input[0], store_values=True)
                self._features[name + '.Q'] = model.Q.detach().cpu()
                self._features[name + '.K'] = model.K.detach().cpu()
                self._features[name + '.V'] = model.V.detach().cpu()
                self._features[name + '.A'] = model.A.detach().cpu()
        return hook

    def reset_features(self):
        self.features = dict()

    def step(self, batch):
        x, y = batch
        output = self.model(x)
        B, C, H, W = x.shape

        if not self.features:
            self.features['input'] = x.view(B, -1)
            self.features['labels'] = y.view(B, 1)
            self.features['output'] = output.detach().cpu()
            for layer in self.layers:
                self.features[layer] = self._features[layer].cpu()
                if layer.find('mhsa.head') >= 0:
                    self.features[layer + '.Q'] = self._features[layer + '.Q']
                    self.features[layer + '.K'] = self._features[layer + '.K']
                    self.features[layer + '.V'] = self._features[layer + '.V']
                    self.features[layer + '.A'] = self._features[layer + '.A']
        else:
            self.features['input'] = torch.concat((self.features['input'], x.view(B, -1)), dim=0)
            self.features['labels'] = torch.concat((self.features['labels'], y.view(-1, 1)), dim=0)
            self.features['output'] = torch.concat((self.features['output'], output.detach().cpu()), dim=0)
            for layer in self.layers:
                self.features[layer] = torch.concat((self.features[layer], self._features[layer].cpu()), dim=0)
                if layer.find('mhsa.head') >= 0:
                    self.features[layer + '.Q'] = torch.concat((self.features[layer + '.Q'], self._features[layer + '.Q']), dim=0)
                    self.features[layer + '.K'] = torch.concat((self.features[layer + '.K'], self._features[layer + '.K']), dim=0)
                    self.features[layer + '.V'] = torch.concat((self.features[layer + '.V'], self._features[layer + '.V']), dim=0)
                    self.features[layer + '.A'] = torch.concat((self.features[layer + '.A'], self._features[layer + '.A']), dim=0)

    def collect_features(self, dm):
        """Collect model features on dataset"""
        self.reset_features()
        for idx, inputs in enumerate(dm):
            self.step(inputs)

    def PCA(self, layers=None, k=3, features=None, embed_index=None):
        """Principal Component Analysis

        Note that embedded vectors are treated independently. That is, for an embedding or attention layer, you have
        something like Q with dimensions (L, d_k), where each row is a separate embedding. We treat each of these
        embeddings as separate samples, as opposed to concatenating them into a longer feature vector. If you wish to
        only consider a single embedding (e.g. the "class token embedding" -> usually embed_idx=0), you may pass in an
        embed_index, which will pick out only that embedding when collecting features.
        """
        if features is None:
            features = self.features

        if layers is None:
            layers = features.keys()

        PCA = dict()
        for layer in layers:
            if layer == 'labels':
                continue

            if layer.find('mhsa') >= 0 or layer.find('embedding') >= 0:
                B, L, d_ = features[layer].shape
                if embed_index is None:
                    f = features[layer].view(B * L, d_).float()
                else:
                    f = features[layer][:, embed_index, :].squeeze(dim=1)
            else:
                f = features[layer].float()

            k = min(f.shape[1], k)
            U, S, V = torch.pca_lowrank(f, q=k)
            projection = torch.matmul(f, V[:, :k])
            approx = torch.matmul(projection, V.T)

            PCA[layer] = dict()
            PCA[layer]['V'] = V
            PCA[layer]['proj'] = projection
            PCA[layer]['acc_var'] = torch.var(approx) / torch.var(projection) * 100

        return PCA

    def PCA_scatter(self, layers=None, k=3, features=None, embed_index=None, PCA=None, save_fig_filename=None):
        assert k == 2 or k == 3, 'k must be 2 or 3'
        if features is None:
            features = self.features
        if layers is None:
            layers = features.keys()
        if PCA is None:
            PCA = self.PCA(layers=layers, k=k, features=features, embed_index=embed_index)

        for layer in layers:
            num_classes = len(torch.unique(features['labels']))
            f = PCA[layer]['proj']
            if k==3:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig, ax = plt.subplots()
            labels = features['labels']
            if f.shape[0] != labels.shape[0]:  # e.g. if we are in an embedding or mhsa layer
                d_ = f.shape[0] // labels.shape[0]
                labels = labels.repeat_interleave(repeats=d_, dim=0)
            if k == 2:
                sm = ax.scatter(f[:, 0], f[:, 1], c=labels, cmap='Spectral')
            else:
                sm = ax.scatter(f[:, 0], f[:, 1], f[:, 2], c=labels, cmap='Spectral')
            plt.title(f'{layer} PCA')
            plt.colorbar(sm, boundaries=np.arange(num_classes+1) - 0.5).set_ticks(np.arange(num_classes))

        if save_fig_filename is None:
            plt.show()
        else:
            plt.savefig(save_fig_filename, transparent=False, facecolor='white')

    def create_gif(self, fig_filenames, save_gif_filename, fps=5):
        frames = list()
        for fn in fig_filenames:
            image = imageio.v3.imread(fn)
            frames.append(image)
        imageio.mimsave(f'{save_gif_filename}', frames, fps=fps)




