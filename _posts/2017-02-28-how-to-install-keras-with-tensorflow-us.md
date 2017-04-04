---
layout: post
title:  "How to Install Keras with Tensorflow using Conda"
date:   2017-02-28 15:00:14 -0700
categories: keras tensorflow conda python
---

[Keras][keras] is an amazing wrapper for [Tensorflow][tensorflow]
(and [Torch][torch]) that makes it simple to start playing with Neural Networks.

Using environment manager like [Anaconda][anaconda] makes life easier.
But sometimes due to different dependencies it takes additional steps to unserstand
how to install needed packages.

I assume that you have `Anaconda` installed.

Since there is no `tensorflow` package in an [Anaconda Package List][anaconda-package-list]
one have to use [conda-forge][conda-forge] - community supported repository of packages.

But as of February 27, 2017 the latest Python version is 3.6 and conda-forge lacks tensorflow
package for that version.

So first of all, let's create a 'tensorflow' environment with the Python version 3.5:

```
conda create -n tensorflow python=3.5
```

Then we make new environment active, install `tensorflow` package from the `conda-forge` channel,
and install available :

```
source activate tensorflow
conda install -c conda-forge tensorflow
conda install -c conda-forge keras=1.0.7
```

## Testing that Tensorflow is working

```
python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

The succesfull output should be:
```
Hello, TensorFlow!
```

(available versions: conda search --override-channels --channel conda-forge  keras)

## Set up Keras

To work with Tensorflow as backend, please make sure that you have the following in the `~/.keras/keras.json` file:

```
{
    "image_dim_ordering": "th",
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow"
}
```

That's it, you are ready to use Keras with Tensorflow!

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
[anaconda]: https://docs.continuum.io/anaconda/
[anaconda-package-list]: https://docs.continuum.io/anaconda/pkg-docs.html
[conda-forge]: https://conda-forge.github.io
[keras]: https://github.com/fchollet/keras
[tensorflow]: https://www.tensorflow.org
[torch]: http://torch.ch