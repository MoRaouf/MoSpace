---
toc: true
layout: post
description: "Word2vec math & code explanation with TensorFlow"
categories: [post]
tags: [Word Embedding, Word2vec, TensorFlow]
title: "Understanding Word Embeddings"
image: "images/WordEmbedding/WEPost.png"
comments: true
featured: true
---
# <u>Word Embedding</u>
## <u>Word2vec</u>

- Word2vec is not a singular algorithm, rather, it is a family of model architectures and optimizations that can be used to learn word embeddings from large datasets.
- There are two main models that help to learn word embeddings:
    - **Continuous bag-of-words model**: predicts the middle word based on surrounding context words. The context consists of a few words before and after the current (middle) word. **This architecture is called a bag-of-words model as the order of words in the context is not important**.
    - **Skip-gram model**: predicts words within a certain range before and after the current word in the same sentence.<br/><br/>

### **Skip-gram and negative sampling**

- While a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of `(target_word, context_word)` where `context_word` appears in the neighboring context of `target_word`.
- The context words for a `target word` is defined by a **window size**. The window size determines the span of words on either side of a `target_word` that can be considered a `context word`.
    
    ![]({{ "images/WordEmbedding/WE0.png" | relative_url }})
    
- At the start of the training phase, we create two matrices – an `Embedding` matrix and a `Context` matrix. These two matrices have an embedding for each word in our vocabulary (So `vocab_size` is one of their dimensions). The second dimension is how long we want each embedding to be (`embedding_size`  , 300 is a common value). These two matrices are for the same words, & are initialized randomly. After training, we discard the `Context` matrix & choose the `Embedding` matrix as our word embedding representation.
    
    ![]({{ "images/WordEmbedding/WE1.png" | relative_url }})<br/><br/>
    

### Objective Function for the Skip-gram

- The training objective of the skip-gram model is to maximize the probability of predicting context words given the target word. For a sequence of words $*w1, w2, ... wT*$, the objective can be written as the average log probability, where `c` is the size of the training context.:
    - $\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\le j \le c,j\ne0} log \space p(w_{t+j}\vert w_t)$
    - In this objective function, we are calculating for each target word $w_t$ the summation of probabilities of each context word $w_{t+j}$ in the context span from $-c\le j \le c$ , & $j\ne0$ so we don’t calculate probability of predicting the same target word as a context word given it as a target. Then we sum over all the words sequence & divide by the total number of words. In this way we want to maximize the output probability as this is like we are calculating a probability for a single word & we want it as maximum as possible. If we manage to get for each target word the maximum probability, lets say more than 0.9, then when we sum over all words probabilities & divide by their number we will get the maximum probability for the objective function & in this way we are minimizing the error & getting better word embedding representations.<br/><br/>

### Context Word Probability $P(w_c \vert w_t)$

- The conditional probability of generating the context word $w_c$ for the given central target word $w_t$ can be obtained by performing a softmax operation on the vectors inner product:
    - $P(w_c \vert w_t) = \frac{exp(u_c^⊤*v_t)}{\sum_{i\in V} exp(u_i^T*v_t)}$ , where $u_c$ is the vector of the context word, $v_c$ is the vector of the target word & $V$ is the length of the vocabulary.
    - Computing the denominator of this formulation involves performing a full softmax over the entire vocabulary words, which are often large ($10^5$ to $10^7$) terms.<br/><br/>

### Gradient Descent

- The key of gradient computation is to compute the gradient of the logarithmic conditional probability for the central target word vector and the context word vector. By definition, if we take the $log$  for both sides, we first have:
    - $log \space P (w_c \vert w_t) = (u_c^⊤ * v_t) − log(\sum_{i \in V} exp({u_i^T} * v_t))$
    - Through differentiation, we can get the gradient of $v_t$ from the formula above by differentiating with respect to $v_t$:
        - $\frac{\partial \space log \space P (w_c \vert w_t)}{\partial v_t} = u_c-\frac{\sum_{j \in V} exp({u_j^T} * v_t) * u_j}{\sum_{i \in V} exp({u_i^T} * v_t)}$
        , we consider only the context words ($j$s) of the current target word in the numerator.
        - $\frac{\partial \space log \space P (w_c \vert w_t)}{\partial v_t} = u_c-\sum_{j \in V}(\frac{ exp({u_j^T} * v_t)}{\sum_{i \in V} exp({u_i^T} * v_t)}) * u_j$
        , we reformat the differentiation by taking out the summation over the context words & their vectors.
        - $\frac{\partial \space log \space P (w_c \vert w_t)}{\partial v_t} = u_c-\sum_{j \in V}P(w_j \vert w_t)*u_j$<br/><br/>

### Negative Sampling

- To generate high-quality embeddings using a high-performance model, we can switch the model’s task from predicting a neighboring word:
    
    ![]({{ "images/WordEmbedding/WE2.png" | relative_url }})
    
- And switch it to a model that takes the input and output word, and outputs a score indicating if they’re neighbors or not (0 for “not neighbors”, 1 for “neighbors”).
    
    ![]({{ "images/WordEmbedding/WE3.png" | relative_url }})
    
- So we switch from the left dataset structure to the right one below:
    
    ![]({{ "images/WordEmbedding/WE4.png" | relative_url }})
    
- When we apply skipgram on a sentence, we generate pairs of `(target_word, context_word)` & these pairs are all positive with target equal to 1 (i.e., we know all the true context words for the target word). If we train a model on these pairs, it will get an accuracy of 100% as we have only one class of output.
- To produce additional skip-gram pairs that would serve as negative samples for training, you need to sample random words from the vocabulary. **Negative samples** are samples of words that are not neighbors.  Our model needs to return 0 for those samples. So we get the following dataset instead:
    
    ![]({{ "images/WordEmbedding/WE5.png" | relative_url }})
    

### Skip-gram with Negative Sampling

- In a nutshell, the two main ideas of this model can be visualized as follows:
    
    ![]({{ "images/WordEmbedding/WE6.png" | relative_url }})<br/><br/>
    

### Word2vec Training

- Before the training process starts, we pre-process the text we’re training the model against. In this step, we determine the size of our vocabulary (we’ll call this `vocab_size`, think of it as, say, 10,000) and which words belong to it.
- Then we have our `Embedding` matrix and `Context` matrix. We initialize these matrices with random values. Then we start the training process. In each training step, we take one positive example and its associated negative examples.
    
    ![]({{ "images/WordEmbedding/WE7.png" | relative_url }})
    
- For the input word, we look in the `Embedding` matrix, & for the context words, we look in the `Context` matrix.
    
    ![]({{ "images/WordEmbedding/WE8.png" | relative_url }})
    
- Then, we take the dot product of the input embedding vector with each of the context embeddings vectors (this is the numerator part in the (Context Word Probability section). Then we apply the softmax for each sample ( for positives & negatives). Then we go through backpropagation to update our matrices.<br/><br/>

### Hyperparameters

- Two key hyperparameters in the word2vec training process are the **window size** and the **number of negative samples**.
- The `Gensim` library default window size is 5 (two words before and two words after the input word, in addition to the input word itself).
- The original paper prescribes 5-20 as being a good number of negative samples. It also states that 2-5 seems to be enough when you have a large enough dataset. The `Gensim` library default is 5 negative samples.<br/><br/>

## <u>TensorFlow</u>

- You can use the `tf.keras.preprocessing.sequence.skipgrams` to generate skip-gram pairs from the `sequence` with a given `window_size` from tokens in the range `[0, vocab_size)`
    ```python
    tf.keras.preprocessing.sequence.skipgrams(
        sequence,
        vocabulary_size,
        window_size=4,
        negative_samples=1.0,
        shuffle=True,
        categorical=False,
        sampling_table=None,
        seed=None
)
```
    - **Returns:** couples, labels; where `couples` are **int pairs** and `labels` are either 0 or 1.
- A [comprehensive chart](https://www.tensorflow.org/tutorials/text/word2vec#summary) of creating skipgram pairs using TensorFlow is given as:
    
    ![]({{ "images/WordEmbedding/WE9.png" | relative_url }})<br/><br/>
    
    
### Embedding through TensorFlow

 ```python
    vocab_size = 10000 
    max_length = 120
    embedding_dim = 16
    #Build the model
    model = tf.keras.Sequential([
        tf.keras.layers**.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)**,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
```

- `input_dim`: Size of the vocabulary in our corpus.
- `output_dim`: Dimension of the dense embedding we want to produce for each word.
- `input_length`:Length of input sequences for each sentence, when it is constant. This argument is required if you are going to connect `Flatten` then `Dense` layers upstream (without it, the shape of the dense outputs cannot be computed).
- **Input shape:** 2D tensor with shape: `(batch_size, input_length)`  —→>>> **[#samples, #words_in_sample_represented_as_`max_length`]**
    - `batch_size` is the number of samples we input to the layer
    - Our input to the embedding layer is a list of samples of length = `vocab_size`  where each sample is a list of words & we limit each sentence to a length of words = `max_length`
    - As each sample has words of `max_length` , then we want each word to be represented as a vector of length = `embedding_dim` instead of being represented as a single number.
    - It’s not necessarily to have in each sentence the same words, so there are no fixed columns header.
- **Output shape:** 3D tensor with shape: `(batch_size, input_length, output_dim)`
    - After the transformation of each word from a single number (from the `word_index`) to a vector of embedding, we can visualize our input matrix to have a 3rd dimension of depth protrudes from each cell that represents a single word embedding vector.
    
    > ***In the below picture, we can think of our input to be [2,5]. 2 samples where each sample has `max_length` of 5 words. After transformation, each word will be represented as an embedding vector of length 3.***
    > 
    
    ![]({{ "images/WordEmbedding/WE10.png" | relative_url }})


<u>**Useful links**</u>:
<br/><br/>
[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
[Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)
[Word2vec](https://stopwolf.github.io/blog/nlp/paper/2020/08/17/word2vec.html)