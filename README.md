# Oak

Oak is an educational package for implementing transformers, and is meant as a
reference implementation for people looking to learn or implement their own
transformer networks, as well as playing around with some of their dynamics.

There are three main architectures covered:
1. **Transformer**, as in [Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)
2. **GPT**, as in [Radford et al. 2018](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
3. **Vision Transformer**, as in [Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf)

Oak used much of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) as a 
reference implementation itself, which is also a great place to start learning the 
transformer architecture. 

# Transformers

[Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf) introduced the Transformer
architecture, which was originally designed as a translation AI. The transformer takes
as input a text sequence in the *source* language, and then iteratively predicts the next 
word in the *target* language by passing in the entirety of the source text, as well as 
all of the target text generated up to that point. That is, the transformer is meant to 
be used iteratively, where it builds up the desired output over time.

[Radford et al. 2018](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) 
introduced Generative Pre-Trained (GPT) models. The main idea is to pre-train a 
transformer encoder, which can then be reused for a variety of tasks by adding 
task-specific heads onto the end of the pre-trained model.

[Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf) introduced the Vision 
Transformer (ViT). The main idea was to replace the text embedding module of the original
transformer with a patch embedding module, which takes in images and breaks them into a 
grid of *patches*, which are then treated as if they were *words* in the original 
transformer. These patches are then passed through a transformer encoder for image 
classification. Other minor alterations, like the inclusion of class embedding tokens and
learnable embeddings, are included in this repo.

At their core, all transformer architectures are based on the 
[Attention](./oak/core/attention.py) Module, which takes an input of dimensions 
($B$, $L$, $d_{model}$) and returns an embedding of dimensions ($B$, $L$, $d_v$), 
where $d_v$ is usually set to $d_{model}$ in practice. The Attention Module is 
based around using an Attention *Kernel* to compute an Attention *Matrix*. 
Common Attention kernels include,

|            Kernel, $A(Q, K)$            |      Common Name      |                      Scientific Name (Citation)                      |
|:---------------------------------------:|:---------------------:|:--------------------------------------------------------------------:|
|             $cosine(Q, K)$              |   Cosine Similarity   |    Cosine Similarity ([Graves](https://arxiv.org/abs/1410.5401))     |
|            $softmax(Q + K)$             |       Additive        |             [Bahdanau](https://arxiv.org/abs/1409.0473)              |
|           $softmax(Q\cdot K)$           |    Multiplicative     |              [Luong](https://arxiv.org/abs/1508.04025)               |
| $softmax(\frac{Q\cdot K^T}{sqrt{d_k}})$ | Scaled Multiplicative | Scaled Dot-Product ([Vaswani](https://arxiv.org/pdf/1706.03762.pdf)) |

In each case the Attention Kernel, $A(Q, K)$, computes the similarity between 
the rows of two matrices, $Q$ and $K$, which it stores in another matrix called 
the Attention Matrix, $A$. This Attention Matrix is used to weight the output 
of the Attention Module, $Z=AV$, where $Z$ is the output of the Attention 
Module, and V is a matrix computed from the input, similarly to Q and K. That 
is, explicitly, Q, K and V are computed from two input matrices, X and Y, as:

$Q = X\cdot W_Q$

$K = Y\cdot W_K$

$V = Y\cdot W_V$

where $W_Q$, $W_K$ and $W_V$ are tunable weight matrices learned via 
backpropagation, and $X$ and $Y$ are the input matrices into the Attention 
Module. The general case, where $X != Y$, is called Cross-Attention, and the
more common case, where $X = Y$ is called Self-Attention.

In practice, each Attention Module is split into multiple *heads* that each
receive a diminished dimension of the original embedding.
That is, if the input into the module is ($B$, $L$, $d_{model}$), the input 
dimension received by each Attention *Head* is ($B$, $L$, $d_{model} / h$), 
where $h$ is the total number of Heads. It is understood that, in practice,
each Attention Module is actually one of many Attention *Heads*, where the
collection of all heads come together in a MultiHeadAttention (MHA) Module, 
which concatenates the output of all comprised heads before passing on the 
output to the next MHA Module, which repeats the process.

That is, in practice, the building blocks of a transformer architecture are
MHA Modules, each comprising $h$ Attention Modules.

# Model Comparison

Comparing the three models, the original transformer is actually the most complicated of 
the models, as it includes both encoder and decoder portions of the network, tying them 
together using cross-attention. Summarizing the major components of each:

|           | Transformer |       GPT        |         ViT          |
|----------:|:-----------:|:----------------:|:--------------------:|
| Embedding |    Text     |       Text       |        Image         |
|   Encoder |      x      |        x         |          x           |
|   Decoder |      x      |                  |                      |
|   Softmax |      x      |        x         |          x           |
|      Task | Translation | *Text Generation | Image Classification |

*Note on GPT: the original paper actually uses the GPT model for a myriad of different
tasks, using task-specific additional heads, which we take as future work here.

Moreover, in terms of the types of Attention used, the original Transformer architecture 
is also the most complicated, as seen below:

|                  | Transformer | GPT | ViT |
|-----------------:|:-----------:|:---:|:---:|
|   Self-Attention |      x      |  x  |  x  |
| Masked-Attention |      x      |  x  |     |
|  Cross-Attention |      x      |     |     |

# Package Components

Oak is meant to be easily examined and hacked for your own purposes. As such, the 
individual package components map directly onto the original architecture diagrams, as 
seen below:

![Package Components](./resources/transformer_package_components.png)
![ViT Package Components](./resources/vit_package_components.png)

At the highest level, each transformer architecture is located in 
[oak.transformers](./oak/transformers):
1. Transformer, as in [oak.transformers.transformer](./oak/transformers/transformer.py)
2. GPT, as in [oak.transformers.gpt](./oak/transformers/gpt.py)
3. ViT, as in [oak.transformers.vision_transformer](./oak/transformers/vision_transformer.py)

Each of these models combines an embedding module (either a 
[text embedding module](./oak/embeddings/text_embedding.py) or an 
[image embedding module](./oak/embeddings/image_embedding.py)), an encoder comprising one 
or more [encoder blocks](./oak/core/encoder_block.py) and, optionally, a decoder comprising 
one or more [decoder blocks](./oak/core/decoder_block.py).





Oak is meant to be easily examined and used for your own hacking purposes. The 
main components of Oak directly follow the components from Transformer architecture 
as described in [Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)
and [Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf):

1. **Embedding** ([oak.core.embedding.Embedding](./oak/core/image_embedding.py)). The input into a transformer is a 
    (B, N, d_model) tensor, where $B$ is the batch size, $N$ is the number of 
    representational vectors, and $d_{model}$ is the embedding dimension of each 
    vector. That is, the Transformer does not actually know if it's receiving images,
    text, audio, or any other domain. It just takes a vector space and transforms it
    into another vector space. The embedding module is responsible for creating this 
    first vector space in a sensible way given the input domain. That is, an embedding
    module converts text or images into a (B, N, d_model) tensor. As such, each input 
    type may use the exact same transformer architecture and just swap out a different
    embedding module.
2. **Scaled Dot-Product Attention** ([oak.core.attention.Attention](./oak/core/attention.py)). 
    Vaswani et al. (2017) uses a dot-product attention module defined by the
    equation $Attention(X, Y) = softmax(\frac{QK^T}{\sqrt{d_k}})\cdot V$, where
    Q, K and V are matrices computed from X and Y:
        
    $Q = XW_k $
    $K = YW_k $
    $V = YW_k $
3. **Multi-head Attention** ([oak.core.mhsa.MultiHeadAttention](./oak/core/mhsa.MultiHeadAttention)). 
    In practice, the inputs into an attention module are split into *h* 
    different "heads". Each head is its own attention module, with a reduced 
    dimensionality (i.e. each head still receives input from each word token /
    image patch, but with embedding dimension $d_{model} / h$ instead of $d_{model}$).

4. **Feed Forward / MLP** ([oak.core.mlp.MLP](./oak/core/mlp.py)). Attention layers 
    are commonly interspersed with feed forward layers. We follow
    [Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf) in using a 
    single-hidden layer MLP as the feed forward component.
5. **Attention Block** ([oak.core.sab.SelfAttentionBlock](./oak/core/encoder_block.py)). Attention 
    layers are coupled with feed-forward layers into "blocks" which get repeated to
    make the network sufficiently deep. Each block pairs an attention layer and feed 
    feed forward layer with a norm layer, and then adds residual connections and
    dropout.
6. **Transformer** ([oak.core.vision_transformer](./oak/core/vision_transformer.py)).
    The Transformer ties an embedding module with a number of attention blocks,
    along with a final MLP head matching the task at hand.


# Reference Notation
Oak uses the following convention throughout:

| Image-based Hyperparameters |                                         Meaning                                         |
|:---------------------------:|:---------------------------------------------------------------------------------------:|
|           C, H, W           | The number of channels (typically either 1 or 3), height and width of the input images. |
|              P              |                 The patch size; i.e. each patch will be P by P pixels.                  |
|              N              |            The number of patches, for image embeddings. $N = (H/P) * (W/P)$             |
|              L              |   Equivalent to seq_len for text-based transformers. For ViT it will be equal to N+1.   |

| Text-based Hyperparameters |                        Meaning                         |
|:--------------------------:|:------------------------------------------------------:|
|         L, seq_len         |      The number of tokens in each input sequence.      |

| Model Hyperparameters |                                                                                                                                                                                                                                   Meaning                                                                                                                                                                                                                                   |
|:---------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      $d_{model}$      |                                                                                                                                                             The embedding dimension of the input rows as they are pass through the network. Also commonly written in the literature as $d_{emb}$ or simply $d$.                                                                                                                                                             |
|         $d_k$         | In the process of creating the Attention matrix, there is a hidden dimension between the queries and the keys; this hidden dimension is d_k. That is, outside of the Attention module changing d_k will have no visible effect (that is, the output dimensions will not change, nor is $d_k$ constrained by $h$ or $d_{model}$ hyperparameters). Increasing d_k, however, will allow the Attention matrix computation to be more expressive / happen in a higher dimension. |
|         $d_v$         |                                                                                                                         The embedding dimension of the Values within the Attention module, which will appear as the final output dimension of every Attention Module (B, L, d_v). That is, $d_v$ defines the expressivity of the Attention modules.                                                                                                                         |
|           h           |                                                                                                                                                                                                           The number of heads in the MultiHeadAttention modules.                                                                                                                                                                                                            |
|      L, seq_len       |                                                                                                                                     The length of the input. For text embeddings it will be the number of tokens in the input sequence, for image embeddings it will be the number of patches plus one (for the classification token).                                                                                                                                      |
|     B, batch_size     |                                                                                                                                                                                                         The number of samples used in a training / inference batch.                                                                                                                                                                                                         |

| Parameters / Computed Values |               Dimensions               |                                                                                  Meaning                                                                                   |
|:----------------------------:|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|              X               | text: (B, L) <br/> image: (B, C, H, W) |             The raw input into the model. For text-based models the input should be a tokenized sequence. For image-based models the input should be an image.             |
|              Y               |                 (B, L)                 | The raw "second input", when cross-attention will be used. For the original Transformer Y is the tokenized source sentence (i.e. the sentence which should be translated). |
| X (Post Embedding Module), E |          (B, L, $d_{model}$)           |                                                                  The input into an encoder/decoder block.                                                                  |
|  Y (Post Embedding Module)   |          (B, L, $d_{model}$)           |                                           The "second" input; when it is seen inside an Attention or MultiHeadAttention Module.                                            |
|            $W_Q$             |          ($d_{model}$, $d_k$)          |                                                           Trainable weight parameters for computing the Queries.                                                           |
|            $W_K$             |          ($d_{model}$, $d_k$)          |                                                            Trainable weight parameters for computing the Keys.                                                             |
|            $W_V$             |          ($d_{model}$, $d_v$)          |                                                           Trainable weight parameters for computing the Values.                                                            |
|              Q               |               (L, $d_k$)               |                                                                             Queries, $Q=XW_Q$                                                                              |
|              K               |               (L, $d_k$)               |                                                                               Keys, $K=XW_K$                                                                               |
|              V               |               (L, $d_v$)               |                                                                              Values, $V=XW_V$                                                                              |
|              A               |                 (L, L)                 |                     The Attention matrix, $A=softmax(QK.transpose / sqrt{d_k})$. Note that for cross-attention the dimensions will be ($L_X$, $L_Y$).                      |
|              Z               |               (L, $d_v$)               |                                                                The output of an Attention Module, $Z = AV$                                                                 |

# Resources

As part of my own learning about transformers, I found the following resources
incredibly valuable, and highly recommend looking at them yourself if you are 
new to transformers, or just looking to learn about them from the experts who 
make them.

### Papers
1. Original Transformer Paper, [Vaswani et al. 2017](https://arxiv.org/pdf/1706.03762.pdf)
2. Visual Transformer, [Dosovitskiy et al. 2020](https://arxiv.org/pdf/2010.11929.pdf)

### Understanding Transformers
1. Lucas Beyer, Transformers. [Slides](https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/edit#slide=id.g31364026ad_3_2), [Talk](https://www.youtube.com/watch?v=EixI6t5oif0&ab_channel=MLTArtificialIntelligence)
2. Lilian Wang, [The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
3. Aritra Gosthipaty and Sayak Paul, [Investigating Transformer Representations](https://keras.io/examples/vision/probing_vits/)

### Open Source
1. Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [HuggingFace](https://huggingface.co/) ([github](https://github.com/huggingface))
3. [OpenAssistant](https://projects.laion.ai/Open-Assistant/blog) ([github](https://github.com/LAION-AI/Open-Assistant))
4. PyTorch Transformer Implementation ([docs](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html), [source](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer))
