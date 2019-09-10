## Learning Visual Dynamics Models of Rigid Objects using Relational Inductive Biases

### Description
Endowing robots with human-like physical reasoning abilities remains challenging. We argue that existing methods often disregard spatio-temporal relations and by using Graph Neural Networks (GNNs) that incorporate a relational inductive bias, we can shift the learning process towards exploiting relations. In this work, we learn action-conditional forward dynamics models of a simulated manipulation task from visual observations involving cluttered and irregularly shaped objects. Overall, we investigate two GNN approaches and empirically assess their capability to generalize to scenarios with novel and an increasing number of objects. The ﬁrst, Graph Networks (GN) based approach, considers explicitly deﬁned edge attributes and not only does it consistently underperform an auto-encoder baseline that we modiﬁed to predict future states, our results indicate how different edge attributes can signiﬁcantly inﬂuence the predictions. Consequently, we develop the Auto-Predictor that does not rely on explicitly deﬁned edge attributes. It outperforms the baseline and the GN-based models. Our results show the sensitivity of GNN-based approaches to the task representation, the efﬁcacy of relational inductive biases and advocate choosing lightweight approaches that implicitly reason about relations over ones that leave these decisions to human designers.

### Paper
https://arxiv.org/abs/1909.03749

### Citing
```
@article{dynamicsmodels,
  title={Learning Visual Dynamics Models of Rigid Objects using Relational Inductive Biases},
  author={Ferreira, Fabio and Shao, Lin and Asfour, Tamim and Bohg, Jeannette},
  journal={submit/2836373},
  year={2019}
}
```

### Project website
https://sites.google.com/view/dynamicsmodels/home
