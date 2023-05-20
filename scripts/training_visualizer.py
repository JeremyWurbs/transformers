import os
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm

from oak import MNIST, VisionTransformer, LightningModel

#torch.set_float32_matmul_precision('medium')

param = {
    'input_dim': [1, 28, 28],
    'num_classes': 10,
    'num_blocks': 1,
    'P': 14,
    'h': 2,
    'd_model': 64
}

visualizer_param = {
    'layers': ['embedding', 'blocks.0.mhsa.heads.0', 'blocks.0.mhsa.heads.1'],
}

dm = MNIST(batch_size=3, num_workers=32, num_val=180, num_train=60000-180)
model = VisionTransformer(**param)
model = LightningModel(model, use_visualizer=True, visualizer_args=visualizer_param)

trainer = Trainer(max_steps=5000, val_check_interval=1, devices=[0])
trainer.fit(model, dm)
trainer.test(model, dm)

pbar = tqdm(['input', 'embedding', 'blocks.0.mhsa.heads.0', 'blocks.0.mhsa.heads.1', 'output'])
for layer in pbar:
    pbar.set_description(f'Processing {layer}')
    output_dir = os.path.join(os.getcwd(), 'output_dir')
    T = len(model.val_PCA)
    for t in tqdm(range(1, T)):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model.model.PCA_scatter(PCA=model.val_PCA[t], layers=[layer], k=2, save_fig_filename=os.path.join(output_dir, f'{layer}_{t}.png'))
    filenames = [f'{os.path.join(output_dir, layer)}_{i}.png' for i in range(1, T)]
    model.model.create_gif(filenames, os.path.join(output_dir, f'{layer}.gif'), duration=30)
