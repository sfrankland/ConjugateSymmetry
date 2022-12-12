# ConjugateSymmetry
Tensorflow code for simulations from Frankland, Webb, Petrov, O'Reilly & Cohen (2019)
https://cogsci.mindmodeling.org/2019/papers/0313/0313.pdf


Human analogical ability involves the re-use of abstract, structured representations within and across domains. Here, we
present a generative neural network that completes analogies
in a 1D metric space, without explicit training on analogy.
Our model integrates two key ideas. 

First, it operates over representations inspired by properties of the mammalian Entorhinal Cortex (EC), believed to extract low-dimensional representations of the environment from the transition probabilities between states. 

Second, we show that a neural network equipped with a simple predictive objective and highly general
inductive bias can learn to utilize these EC-like codes to compute explicit, abstract relations between pairs of objects. The
proposed inductive bias favors a latent code that consists of
anti-correlated representations. The relational representations
learned by the model can then be used to complete analogies
involving the signed distance between novel input pairs (1:3
:: 5:? (7)), and extrapolate outside of the network’s training
domain. As a proof of principle, we extend the same architecture to more richly structured tree representations.
