---
layout: post
title:  "How to Install Keras with Tensorflow using Conda"
date:   2017-02-28 15:00:14 -0700
categories: python keras
tags: keras tensorflow conda python
---

*Updated 28 Jun 2017.*

[Keras][keras] is an amazing wrapper for [Tensorflow][tensorflow]
(and [Torch][torch]) that makes it simple to start playing with Neural Networks.

Using environment manager like [Anaconda][anaconda] makes life easier.
But sometimes due to different dependencies it takes additional steps to unserstand
how to install needed packages.

I assume that you have `Anaconda` installed.

Since there is no `tensorflow` package in an [Anaconda Package List][anaconda-package-list]
one have to use [conda-forge][conda-forge] - community supported repository of packages.

~~But as of February 27, 2017 the latest Python version is 3.6 and conda-forge lacks tensorflow
package for that version.~~

So first of all, let's create environment with the Python, and name it a 'tf'. 
I also advice to install `pandas`, `matplotlib`, and `jupyter` packages for data manipulation
and visualization of the result.

```
conda create -n tf python=3 pandas matplotlib jupyter keras tensorflow
```

Then we make new environment active:

```
source activate tf
```

## Testing that Tensorflow is working

```
python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

There would be warnings that `The TensorFlow library wasn't compiled to use <...> instructions, ...`.
That is ok. We don't want to build libraries from the the source code here.

The succesfull output should be:
```
Hello, TensorFlow!
```

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
Let's do some "Hello, World!" [handwritten digits recognition]({% post_url 2017-04-21-tutorial-on-using-keras-for-digits-recognition %}).

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
[anaconda]: https://docs.continuum.io/anaconda/
[anaconda-package-list]: https://docs.continuum.io/anaconda/pkg-docs.html
[conda-forge]: https://conda-forge.github.io
[keras]: https://github.com/fchollet/keras
[tensorflow]: https://www.tensorflow.org
[torch]: http://torch.ch