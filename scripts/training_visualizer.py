import os
from pytorch_lightning import Trainer

from oak import MNIST, Visualizer, VisionTransformer, LightningModel

#torch.set_float32_matmul_precision('medium')

param = {
    'input_dim': [1, 28, 28],
    'num_classes': 10,
    'blocks': 1,
    'P': 14,
    'h': 2,
    'd_model': 64
}

visualizer_param = {
    'layers': ['embedding', 'blocks.0.mhsa.heads.0', 'blocks.0.mhsa.heads.1'],
}

dm = MNIST(batch_size=32, num_workers=1)
model = VisionTransformer(**param)
model = LightningModel(model, use_visualizer=True, visualizer_args=visualizer_param)

trainer = Trainer(max_epochs=3, val_check_interval=0.5)
trainer.fit(model, dm)
trainer.test(model, dm)

for layer in ['input', 'model.embedding', 'model.blocks.0.mhsa.heads.0', 'model.blocks.0.mhsa.heads.1', 'output']:
    output_dir = os.path.join(os.getcwd(), 'output_dir')
    for t in range(1, 6):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model.model.PCA_scatter(PCA=model.val_PCA[t], layers=[layer], save_fig_filename=os.path.join(output_dir, f'{layer}_{t}.png'))
    filenames = [f'{os.path.join(output_dir, layer)}_{i}.png' for i in range(1, 6)]
    model.model.create_gif(filenames, os.path.join(output_dir, f'{layer}.gif'), fps=5)
