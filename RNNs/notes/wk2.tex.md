# Week 2 - Intro to word embeddings

Word embeddings captures analogies like that man maps to king, and woman maps to queen.

Word embeddings allow NLP models with reltively small datasets.

Embeddings are a matrix of $(word \times concept)$ which captures the relatedness of a words to concepts. Eg, Gender, food, action, expensive.

t-SNE takes a high-dimensional space allows visualisation in 2D.

[Laurens van der Maaten, Geoffrey Hinton, 2008: Visualizing Data using t-SNE](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

1. Use dense word embeddings as input to RNNs rather than the sparse one-hot encodings.
1. Use transfer learning to new task with smaller training set
1. Finetune word embeddings with the new data if the dataset is "large enough".

Word embeddings are very useful when there is a smaller-sized labelled training set.  Has been used in: name entity recogniton, parsing, text summarisation, co-reference resolution. Less useful for langage modelling machine translation especially given a large dataset.

Whereas the encoding an never-seen-before face as a 128-dimensional vector, word embeddings use a pre-defined dictionary size of say 10,000 words, and learn an embedding for each of the words.

### Using word embeddings

[Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig, 2013: Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N13-1090)  1/5 difficulty

Embeddings have the property such that:

$e_{man} - e_{woman} \approx e_{king} - e_{queen}$

King is to $x$ as man is to woman:

$\underset{x}{\mathrm{argmax}}\ similar(e_{x} \approx e_{king} - e_{man} + e_{woman})$

This gives a 30-75% accuracy on predicting the exact expected word.

To find the vector with the highest similarity, either the $cos$ of the vectors can be used (seeking the highest value), or the euclidean distance (seeking the lowest value). $cos$ is used more often, and normalises the lengths of the vectors.

### Learning word embeddings

1. The last $n$ words can be used for learning. The embedding of each is obtained, then fed to a NN with idden layer's size is $n \times \text{embedding size}$. The output is a softmax of the size of the dictionary.

2. Context can also be the previous $n$ words and the succeeding $m$ words, or a nearby single word (skip gram).

For a language model, use the last $n$ words as context. For learning word embeddings, the contexts of the 2nd point can be used also.

### Word2Vec (skip-gram and CBOW)

[Mikolov et al, 2015. Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) 1/5

Select a random target word nearby or with $\pm [5,10]$ words of the context word, and try to predict it.

When you define the window size parameter, you only configure the maximum window size. The actual window size is randomly chosen between 1 and max size for each training sample, resulting in words with the maximum distance being observed with a probability of 1/c while words directly next to the given word are always(!) observed.

This is a very difficult task to get right, but the goal isn't to predict nearby words, but to learn good word embeddings.

The model is fed the embedding of the context word $x$, to a single softmax which predicts the target word $y$.

The problem with this is the computational requirements of the softmax denominator: $\sum_{i=1}^V exp(\theta_t^T e_c + b_t)$. Given vocab size $V$ can be 1 billion words this is extreme, and still onerous for only 10K words.

A heirarchical classifier built from a tree of binary classifiers can be used instead, and this is $\mathcal O log|V|$ where $V$ is the vocab size.

An unbalanced tree can be used with the more common word leaves closer to the top requiring fewer traversals.

To prevent over-training of very common words, and to ensure more training of very infrequent words, a non-uniform selection of context word is used.

The Continous Bag Of Words (CBOW) was also described in the paper. It uses a sliding window of context words to predict the target word.

* Skip-gram: works well with small amount of the training data, represents well even rare words or phrases.
* CBOW: several times faster to train than the skip-gram, slightly better accuracy for the frequent words

### Negative sampling

The softmax function in the skip-gram is still expensive to compute even with a heirarchical classifier.

[Mikolov et al, 2013. Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)  1/5

In this method, we build a table of (this word), (other word) and (is context).

If (other word) would have been picked as a pair in the skip-gram, then (is context) gets 1, else it is a 0.

Apart from the (is context = 1) words sampled from the $\pm 10$ context words, pick $k$ words at random from the dictionary to fill out (other words) with (is context = 0).

We pick 1 (is context = 1) word, and then $k$ words where (is context = 0).

Some randomly picked words with (is context = 0) from the dictionary may have been part of (this word)'s surrounding context. This is ok.

Choose $k \in [5,20]$ for small datasets, and $k \in [2,5]$ for larger datasets.

Given:

$c$ = context word (this word)  
$t$ = target word (other word)  
$y$ = label $\in {0, 1}$  
$e_c$ = the embedding of $c$  
$\theta_t$ = parameter vector (one logit for each target word)

Then this turns into a logistic regression:

$P(y=1|c,t) = \sigma(\theta_t^Te^c)$

The network structure is simple: a single layer of $V$ (vocab size) logits with $e_c$ is input.

Instead of training all $V$ logits, only the $1$ positive example, and $k$ negative example logits are trained on each iteration.

#### Selecting negative examples

If selecting based on the frequency of occurance, then the most common words will be excessively represented. If using $1 \over {|V|}$, less common words will not be trained much.

Through experiment, the authors chose: $\displaystyle \frac {f(w)^{3/4}} {\sum_{j=1}^V f(w)^{3 / 4}}$, where $f(w)$ is the probability of the word appearing based on frequency of occurance.

### GloVe

While skip-grams with negative sampling are used more commonly, GloVe has a following due to its simplicity.

[Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) 3/5

Let $X_{ij}$ be the number of times that the word $i$ appears in the context of the word $j$.

Using "within n words" as the definition of context, then $X_{ij} = X_{ji}$ (a symmetrical relationship). If the context is "the word preceeding", then not.

The algorithm:

$ \displaystyle min \sum_{i=1}^V \sum_{j=1}^V (\theta^T_i e_j      -\log X_{ij})^2$

This learns vectors $\theta_i$ and $e_i$ such that their inner product is a good predictor of how many times word $i$ appears in the context of $j$.

$log(0)$ is undefined as $-\infty$, so we won't sum where $X_{ij} = 0$.

We add an weighting term, $f(X_{ij})=0$ iff $X_{ij} = 0$, and use the convention that $0 \log 0 = 0$.

$ \displaystyle min \sum_{i=1}^V \sum_{j=1}^V f(X_{ij}) (\theta^T_i e_j + b_i + b'_j -\log X_{ij})^2$

$b_i$ and $b'j$ come from the weighting function and are necessary to have the term = 0.

The weighting factor $f(X_{ij})$ gives more weight for less commonly appearing words while demoting the weight for stock words (the, of, a, is).

Given a symmetric definition of target and context words, $\theta_i$ and $e_j$ are the same.  So, initialise $\theta$ and $E$ uniformly random, run gradient descent, then take $ \displaystyle e_w^{\text{(final)}} = \frac {\theta_w + e_w} {2}$.

Individual components of the embeddings generated by GloVe may not represent a single variable like they do in the other methods, but the parallelogram form of:

King is to $x$ as man is to woman: $\underset{x}{\mathrm{argmax}}\ similar(e_{x} \approx e_{king} - e_{man} + e_{woman})$

will still work.


### Sentiment analysis

Word embeddings allow sentiment predictors to be trained with modestly sized labelled datasets of about 10-100K examples.

Simple: Sum or average the embedding of each word, then pass to a softmax classifier.  Problem: lacks information coming from word order

Better: Output of RNN feeding in each word sequentially, then softmax on the output.

# Debiasing word embeddings

[Bolukbasi et al. 2016. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520) 2/5

Word embeddings reflect the biases inherent in the text used to train them.

ML algorithms are influencing colledge admissions, jobs, loan applications, and criminal sentencing guidelines.


1.  Find the directionality of the bias:

    Average related pairs: she - he, grandmother - grandfather, he - him.

    If the bias is across more than one axis, singular value decomposition (SVD) is used (similar ideas to PCA).

1. Neutralise: for every word that is not definitional, project to the mid-point to get rid of the bias.

1. Centre gendered pairs around the gender axis mid-point, by shifting them both equally
