## Attention

The purpose of attention is to calculate context vectors $z^{i}$ by taking weighted sums of the input vectors $x^{i}$ corresponding to specific input tokens.

## Simplified self-attention

In self-attention, our goal is to calculate context vectors $z^{i}$ for each $x^{i}$. A context ector can be interpretted as an enriched embedding vectorby incorporating information from all other elements in the sequence. LLMs need to understand the relationship and relevance of words in a sentence to each other.

We determine attention scores by computing the dot product between the query token and each input token. The attention score vector is then normalized for interpretation and maintaining training stability. In practice, it's more cimmon to use the softmax function for normalization (better extreme values management, ensures positive attention weights, weights can be interpretable as probabilities). Finally, calculate the context $z^2$ by multilying the embedded input tokens $x^i$ with the corresponding attention weights and then summing the results. 

Self-attention is also called a scaled dot-product attention.

## Self-attention

Three matrices are used to project the embedding tokens into query vectors $q^{i} = W_q x^{i}$ (search: token the model tries to understand), key vectors $k^{i} = W_k x^{i}$ (store: indexing), and value vectors $v^{i} = W_v x^{i}$ (retrieve: value in a key-value pair) via matrix multiplication. The size of the matrices determines the size of the output embedding vectors, which can be different from the one of the input embedding vector $x^{i}$ but is ussually the same in GPT-like models.

> In self-attention, the matrices $W_q$, $W_k$, and $W_v$ are typically shared across all tokens within a given layer of the model. It allows the model to capture relationships and dependencies between different tokens in the input sequence, enabling it to perform effective self-attention operations.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/15.webp" width="1000px">
<br/><br/>

1. Convert all inputs $x^{i}$ into query $q^{i}$, key $k^{i}$ and value $v^{i}$ vectors. Select one of the query vactors $q^{T}$ as a target one for the example. 

2. Compute the unnormalized attention scores $A_{sc}$ for every query.

$$ \Omega = A_{sc} = QK^{T}$$

3. To compute attention weights $A_w$, we scale the attention scores by the square root of the embedding dimension, then apply the softmax function for normalization.

$$ A_{w} = softmax\left(\frac{A_{sc}}{\sqrt{d_k}}\right) = softmax\left(\frac{QK^{T}}{\sqrt{d_k}}\right)$$

> Scaling is necessary to avoid vanishing gradients. Large dot products can result in small gradients due to softmax function applied to them. As dot product increases, softmax behaves more like a step-function, resulting in nero zero gradients.

4. Compute the context vector as a weighted sum over the value vectors with corresponding attention weights

$$ A = A_w V = softmax\left(\frac{QK^{T}}{\sqrt{d_k}}\right) V $$

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/17.webp" width="600px">

Self-attention involves trainable matrices $W_q$, $W_k$ and $W_v$. As the model is exposed to data, it adjusts these trainable weights.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/18.webp" width="600px">

## Causal attention, or masked attention.

The causal mechanism prevents the model from accessing future infomation. For many LLM tasks you want the self-attention mechanism to consider only the tokens that appear prior to the current position when predicting the next token in a sequence. Pratically, we should mask out the attention weights above the diagonal, and we normalize the nonmasked attention weights such that the attention weights sum to 1.

1. Compute attention weights using self-attention.

2. Multiply the attention weights matrix $A$ with the lower triangle mask matrix of the same shape to zero-out the values above the diagonal.

3. Renormalize the new attention weights to sum up to 1 again deviding each element by the sum of each row.

> **Note:** if the mask were applied after softmax, it would disrupt the probability distribution created by softmax. Masking after softmax would require re-normalizing the outputs to sum to 1 again, which complicates the process and might lead to unintended effects

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/19.webp" width="600px">
<br></br>

**Computational trick**

The masked attention can be implemented more efficiently if we use the following trick. Mask the unnormalized attention scores above the diagonal with negative infinity values $-Inf$ before they enter the softmax function.

**Dropouts**

Dropouts randomly selects hidden layer units that are ignored **during training** (only). It helps to prevent overfitting. Dropout is typically applied at two specific times: after calculating the attention weights or after applying the attention weights to the value vectors. To compensate for the reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of $1/(1-r_{drop})$ where $r_{drop}$ is a dropout rate and ensure that the average influence of the attention mechanism remains
consistent during both the training and inference phases.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/22.webp" width="600px">

## Multi-head attention

The concept of multi-head attention is to perform the attention mechanism multiple times in parallel, using different linear projections that are learned. This means creating multiple instances of self-attention, each with its own weights, and then combining their outputs. 

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/25.webp" width="600px">