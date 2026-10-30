# Building Perceptron — Theory and Research

Reference notes for the project *Building Perceptron* (La Plateforme).
These notes answer the written questions of the brief and serve as the source
material for the slide presentation, the notebook introduction and the README.

---

## Part 1 — Introduction to Deep Learning



### 1.1 Machine Learning and Deep Learning: definitions

**Artificial Intelligence** is the broadest term: any technique that makes a machine
perform a task we would call intelligent. **Machine Learning** is a subfield of it, and
**Deep Learning** is a subfield of Machine Learning. They are nested, not competing ideas:

> Artificial Intelligence ⊃ Machine Learning ⊃ Deep Learning

#### Machine Learning

Machine Learning is the study of programs that get better at a task by being exposed to
data, instead of following rules written by hand. The standard formal definition is Tom
Mitchell's (*Machine Learning*, McGraw-Hill, 1997):

> "A computer program is said to learn from experience E with respect to some class of
> tasks T and performance measure P, if its performance at tasks T, as measured by P,
> improves with experience E."

The term goes back to Arthur Samuel, who used it in 1959 for a checkers program that
improved by playing games against itself.

Applied to this project: **E** is the 569 patient records, **T** is classifying a tumour
as malignant or benign, and **P** is how often that classification is correct.

#### Deep Learning

Deep Learning is the part of Machine Learning built on artificial neural networks with
several layers stacked between the input and the output. "Deep" refers to that depth —
the number of layers — and nothing more.

What makes depth valuable is **representation learning**. In classical Machine Learning,
a human decides which measurements the model is allowed to see: our dataset exists only
because researchers chose to measure radius, texture, concavity and seven other
properties from cell images. In Deep Learning, the network receives the raw data — the
image itself — and discovers its own useful features, layer after layer: edges first,
then shapes, then structures.

#### Where our Perceptron sits

The Perceptron is a single artificial neuron: one layer, no hidden layers. It is the
historical ancestor of Deep Learning, not an example of it. Understanding it is how you
understand what every single neuron inside a deep network is doing.

**Sources for this section**

- Mitchell, T. (1997). *Machine Learning*. McGraw-Hill, chapter 1 — the formal definition
  of learning (E, T, P). Reproduced verbatim in course material at the
  [University of Oxford](https://www.cs.ox.ac.uk/people/varun.kanade/teaching/ML-AIMS-MT2018/slides/slides0.pdf)
  and [ISTA](https://cvml.ista.ac.at/courses/SML_S16/ml2016-01.pdf).
- Samuel, A. L. (1959). "Some Studies in Machine Learning Using the Game of Checkers."
  *IBM Journal of Research and Development*, 3(3), 210–229 — the origin of the term.
  [Full text, MIT](https://people.csail.mit.edu/brooks/idocs/Samuel.pdf)
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553),
  436–444 — the authoritative definition of Deep Learning, by its three Turing Award
  laureates. [doi:10.1038/nature14539](https://doi.org/10.1038/nature14539)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press,
  chapter 1 — representation learning and the AI ⊃ ML ⊃ DL nesting.
  [Free online](https://www.deeplearningbook.org/)

  


### 1.2 How they differ

Both learn from data, so the real differences are practical: what they need, what they
cost, and what you can read back out of them.

| Criterion | Classical Machine Learning | Deep Learning |
| --- | --- | --- |
| **Who chooses the features** | A human expert designs them | The network learns them from raw data |
| **Data it suits** | Structured, tabular data: rows and columns with named meanings | Unstructured data: images, audio, text, video |
| **Amount of data needed** | Works from a few hundred examples | Usually tens of thousands or more |
| **Computing power** | An ordinary CPU, seconds to minutes | Normally a GPU, hours to weeks |
| **Number of parameters** | Tens to a few thousand | Millions to billions |
| **Interpretability** | Often readable: coefficients, tree splits, feature weights | Largely opaque; needs separate explanation tools |
| **Behaviour as data grows** | Performance tends to plateau | Performance tends to keep improving |

#### The one difference that produces all the others

Every row follows from the first. Once you ask a network to discover its own features
instead of being handed them, it needs far more parameters to represent them, which needs
far more data to fit those parameters, which needs far more computing power — and the
features it invents are internal representations with no name in our language, which is
why the result is hard to read.

So "Deep Learning is more powerful" is the wrong summary. The accurate one is: Deep
Learning moves the work of designing features from the human to the machine, and pays for
it in data, computation and transparency.

**Sources for this section**

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553),
  436–444 — hand-engineered features versus learned representations, and the data and
  computation that learning them requires.
  [doi:10.1038/nature14539](https://doi.org/10.1038/nature14539)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press,
  chapter 1 — depth, parameter counts and scaling with data.
  [Free online](https://www.deeplearningbook.org/)
- Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes
  decisions and use interpretable models instead." *Nature Machine Intelligence*, 1,
  206–215 — the interpretability row, argued specifically for high-stakes domains such
  as healthcare. [arXiv:1811.10154](https://arxiv.org/abs/1811.10154)



### 1.3 When to use one rather than the other
There is no general winner. The choice is settled by four questions about the problem in
front of you.

**1. What shape is the data?**
Tabular data — rows of named measurements — is classical Machine Learning's home ground.
Images, audio, text and video are Deep Learning's. This is the first question to ask, and
it usually decides the matter on its own.

**2. How much of it is there?**
A deep network has millions of parameters to fit. Given a few hundred examples it will
memorise the training set instead of learning from it. Classical models, with far fewer
parameters, work comfortably at that scale.

**3. Does anyone need to understand the decision?**
In medicine, credit and law, a prediction nobody can explain may be unusable whatever its
accuracy. Rudin argues that in high-stakes settings an interpretable model should be the
first choice, rather than a black box explained after the fact.

**4. What can you afford?**
Deep Learning normally needs a GPU and hours to weeks of training. Classical models train
on a laptop in seconds. On a small problem that difference is not worth paying.

#### The evidence against reaching for Deep Learning by default

Deep Learning dominates unstructured data. On tabular data the picture is different.
Grinsztajn, Oyallon and Varoquaux benchmarked both families across many tabular datasets
and found that tree-based models still outperform deep networks there — because tabular
features are irregular, unevenly scaled and often uninformative, which suits decision
trees and penalises neural networks.

#### Applied to this project

Our data is tabular: 569 rows, 30 numeric columns with names a doctor can read. It is
small. The domain is medical, so interpretability matters. It trains in milliseconds on a
CPU. All four questions point to classical Machine Learning — which is where the
Perceptron sits.

It is worth noticing *why* the data looks like this. The 30 features are the output of a
feature-engineering step someone already performed: researchers took digitised images of
fine needle aspirates of breast masses and measured radius, texture, concavity and seven
other properties of each cell nucleus. Had we been handed the original images instead,
that step would be ours to do — and taking over that step is precisely what Deep Learning
exists to do.

**Sources for this section**

- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). "Why do tree-based models still
  outperform deep learning on typical tabular data?" *NeurIPS 2022, Datasets and
  Benchmarks Track* — the benchmark evidence on tabular data.
  [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
- Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes
  decisions and use interpretable models instead." *Nature Machine Intelligence*, 1,
  206–215 — interpretability as a requirement, not a preference, in high-stakes domains.
  [arXiv:1811.10154](https://arxiv.org/abs/1811.10154)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press,
  chapter 5 — capacity, overfitting, and why large models need large datasets.
  [Free online](https://www.deeplearningbook.org/)
- Dua, D., & Graff, C. *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine
  Learning Repository — the features are computed from digitised images of fine needle
  aspirates. [Dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)



### 1.4 Three applications of Deep Learning

The three applications below are each documented in a peer-reviewed publication. Together
they show Deep Learning doing the same job in three different fields: discovering useful
features in raw, unstructured data that no human could specify in advance.

#### 1. Protein structure prediction — AlphaFold

**The problem.** A protein is a chain of amino acids that folds into a precise
three-dimensional shape, and that shape determines what the protein does. Predicting the
shape from the sequence alone had been an open problem in biology for roughly fifty years.
Determining a single structure experimentally could take years of laboratory work.

**What Deep Learning did.** AlphaFold predicts a protein's 3D structure from its amino
acid sequence at an accuracy competitive with experimental methods. DeepMind has since
published predicted structures for over 200 million proteins — effectively every protein
known to science.

**Why it required Deep Learning.** Nobody knew in advance which patterns in a sequence
determine folding, so there was no list of features to hand a classical model. The network
had to find them itself.

**Recognition.** The 2024 Nobel Prize in Chemistry was awarded jointly to Demis Hassabis
and John Jumper for this work, sharing the prize with David Baker.

#### 2. Medical diagnosis from images — diabetic retinopathy screening

**The problem.** Diabetic retinopathy is a leading cause of blindness and is largely
preventable when caught early. Screening requires a trained specialist to examine a
photograph of the retina, and in many regions there are not enough specialists to screen
everyone at risk.

**What Deep Learning did.** A network trained on roughly 128,000 retinal photographs
reached sensitivity and specificity comparable to a panel of board-certified
ophthalmologists.

**Why it required Deep Learning.** The input is a raw photograph. The signs that matter —
microaneurysms, small haemorrhages — are tiny, varied in appearance, and extremely hard to
express as numeric rules.

**Why this example matters here.** This is what our project would be if we had been handed
images instead of 30 measured columns. The contrast is the lesson: raw images call for
Deep Learning; a table of measured features does not.

#### 3. Language — the Transformer and large language models

**The problem.** Translating, summarising or answering in natural language requires
tracking relationships between words that may be far apart. Earlier methods read text
word by word and lost track over distance.

**What Deep Learning did.** The Transformer architecture, published in 2017, replaced
sequential reading with an *attention* mechanism that weighs every word against every
other word simultaneously. It is the architecture underneath essentially every modern
large language model.

**Why it required Deep Learning.** The meaning of a sentence is not a list of measurable
properties. Whatever carries it has to be learned from the text itself.

**Sources for this section**

- Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold."
  *Nature*, 596, 583–589.
  [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) ·
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/34265844/)
- The Nobel Prize in Chemistry 2024 — awarded to David Baker, Demis Hassabis and John
  Jumper. [nobelprize.org](https://www.nobelprize.org/prizes/chemistry/2024/summary/)
- Gulshan, V., et al. (2016). "Development and Validation of a Deep Learning Algorithm for
  Detection of Diabetic Retinopathy in Retinal Fundus Photographs." *JAMA*, 316(22),
  2402–2410.
  [doi:10.1001/jama.2016.17216](https://doi.org/10.1001/jama.2016.17216) ·
  [Google Research](https://research.google/pubs/development-and-validation-of-a-deep-learning-algorithm-for-detection-of-diabetic-retinopathy-in-retinal-fundus-photographs/)
- Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information
  Processing Systems 30 (NeurIPS 2017)*.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) ·
  [NeurIPS proceedings](https://proceedings.neurips.cc/paper/7181-attention-is-all)
- Google *AI Experiments* — the collection suggested in the brief.
  [experiments.withgoogle.com](https://experiments.withgoogle.com/collection/ai) 
  

---

## Part 2 — The Perceptron

### 2.1 What is a Perceptron? The biological neuron and the artificial one

### 2.2 The mathematical function and its terms

### 2.3 The learning rule

### 2.4 The activation function

### 2.5 The training process

### 2.6 The limitations of the Perceptron

---

## Bibliography