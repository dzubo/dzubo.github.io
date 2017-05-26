---
layout: post
title:  "Tuning hyperparameters in neural network using Keras and scikit-learn"
date:   2017-05-25 17:00:11 -0700
categories: machine-learning
tags: keras tensorflow python MNIST "neural networks" "machine learning"
---

In the previous post, we trained a
[neural network with one hidden layer containing 32 nodes]({% post_url 2017-04-21-tutorial-on-using-keras-for-digits-recognition %}).

The accuracy of predictions on the test set, provided by the [Kaggle competition][kaggle-digits] was 0.96.

Can we do better?

Yes, we can. How? To put it simply - we need another model that will give higher accuracy on the test set.
What that 'other' model can be? It can be something completely different, like Random Forest model.

But let's say that we want to use the same class of the models - neural networks. Even more,
let's say we want to use the same architecture of the network - one hidden layer neural network.

If you want, you can go directly to the [notebook with working tuning example][nn-tuning-notebook].

## The problem

How can we define that class of the models, and more important - how can we choose the best model from that class?

The answer is what called [hyperparameter optimization][hyperparameter-optimization-wiki].
Any parameter that changes the properties of the model directly, or changes the training process can be used as hyperparameter
to optimize the model on.

A simple example here - the number of the nodes in the hidden layer. For the starting example, I just used the number `32`.
But what if some another number will work better? 

So, can we just take another number of the nodes, say '64', fit the model, **check the accuracy on the test set**,
and make a conclusion which model is better, right? And then take another number of the nodes, fit the model,
get the accuracy, compare it to the previous best result, remember it, and so on.

**No**. This way at some point we can come up with `1.00` accuracy for the test set, and we did nothing here but [overfitted][wk-overfitting]
the test set. 

So the choice of the model should be based on the training dataset only. How can we choose the best model in this case,
if doing multiple fitting of different models on the training dataset also will lead us to the overfitting?

The answer to this question is cross-validation.
The idea is what we split training dataset into two parts - training set and validation set.
Then we fit the model and test accuracy on the validation set. Then we do another split (randomly, for example),
repeat training and getting accuracy on the validation set.  
This way we will have some statistics of the accuracy, measured on the unseen data. 
And now we can use this metric, often called cross-validation accuracy, to choose the best model from the class of the models.
After the choice of hyperparameters' values, we should refit the model on the full training dataset. And use this one model
to predict the outcome for the test dataset.

The simplest way to do hyperparameters optimization - is 'grid search', which is basically a process of checking the cross-validation
accuracy for manually selected subset of hyperparameters.

## The code

Let's tune the model using two parameters: the number of the nodes
in the hidden layer and learning rate of the optimizer used for neural network training.

For the sake of the current tutorial, it would be enough to say that learning rate defines how much weights of the neural network changes
due to errors observed in the output layer for the current training cycle. Let's use fixed number of epochs and size of the batch.

```python
nodes = [32, 64, 128, 256, 512] # number of nodes in the hidden layer
lrs = [0.001, 0.002, 0.003] # learning rate, default = 0.001
epochs = 15
batch_size = 64
```

A very famous library for machine learning in Python [scikit-learn][scikit-learn] contains grid-search optimizer:
`[model_selection.GridSearchCV][GridSearchCV]`. It takes estimator as a parameter, and this estimator must have 
methods `fit()` and `predict()`. See below how ti use GridSearchCV for the Keras-based neural network model.

```python
def build_model(nodes=10, lr=0.001):
    model = Sequential()
    model.add(Dense(nodes, kernel_initializer='uniform', input_dim=784))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = optimizers.RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return(model)

model = KerasClassifier(build_fn=build_model, epochs=epochs,
                        batch_size=batch_size, verbose=0)
```

Let's define our parameters grid:

```python
param_grid = dict(nodes=nodes, lr=lrs)
param_grid
```

```python
{'lr': [0.001, 0.002, 0.003], 'nodes': [32, 64, 128, 256, 512]}
```

`refit=True` for retraining the best model on the whole training dataset.  
Also, I'm using `verbose=2` to see how the process goes and to estimate the time needed to complete the search.

```python
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,
                    n_jobs=1, refit=True, verbose=2)
grid_result = grid.fit(X, Y)
```

The output would be something like that:

```
Fitting 3 folds for each of 15 candidates, totalling 45 fits
[CV] lr=0.001, nodes=32 ..............................................
[CV] ............................... lr=0.001, nodes=32, total=  10.9s
[CV] lr=0.001, nodes=32 ..............................................
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   11.4s remaining:    0.0s
[CV] ............................... lr=0.001, nodes=32, total=  11.2s
[CV] lr=0.001, nodes=32 ..............................................
...
[CV] lr=0.003, nodes=512 .............................................
[CV] .............................. lr=0.003, nodes=512, total=  41.2s
[Parallel(n_jobs=1)]: Done  45 out of  45 | elapsed: 16.9min finished
```

### A couple of notes

1. I'm working on MacOS, and whenever I chose `n_jobs=2` or more, the Jupyter Notebook just froze forever.
So I'm going with the `n_jobs=1`. That's not a problem, as the training of the model is already programmed in a way that
utilizes multiple cores of the machine.
2. While using GridSearchCV it's impossible, or at least extremely hard to organize storage of the training history for every
run inside cross-validation. I tried but wasn't successful at that. The hope is that [callbacks][keras-callbacks] can be used,
but there is no way to tell inside a callback what split the use at the moment. And the history generated by different splits during CV
rewrites itself.

This is how you can double chech performance of different hyperparameters:

```python
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

Making predictions and saving them for submission:

```python
pred_classes = grid.predict(test_images.values)
pred = pd.DataFrame({'ImageId': range(1, len(pred_classes)+1), 'Label': pred_classes})
pred.to_csv('data/output/subm10-256-lr002.csv', index=False)
```

## The result

The accuracy (public leaderboard score) for the model with `nodes=256, lr=0.002` is `0.978`.

The code for this tutorial is in [the Jupyter Notebook][nn-tuning-notebook].

## Can we do better?

The next steps can be using other machine learning algorithms and/or ensemble the results in one meta-model.


[hyperparameter-optimization-wiki]: https://en.wikipedia.org/wiki/Hyperparameter_optimization\
[wk-overfitting]: https://en.wikipedia.org/wiki/Overfitting
[scikit-learn]: http://scikit-learn.org/
[GridSearchCV]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

[nn-tuning-notebook]: https://github.com/dzubo/digit-recognizer/blob/master/NN-tuning.ipynb
[kaggle-digits]: https://www.kaggle.com/c/digit-recognizer
[kaggle-digits-data]: https://www.kaggle.com/c/digit-recognizer/data
[github-digits]: https://github.com/dzubo/digit-recognizer/