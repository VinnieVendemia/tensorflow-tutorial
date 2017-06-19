# TensorFlow Tutorial

Following the instructions [here](https://www.tensorflow.org/get_started/get_started) to learn a bit more about tensor flow.


## Setup

[This](https://github.com/tensorflow/tensorflow/issues/5478) post helped solve some of my issues when getting setup on my mac.


I had to do the following in a console: 

```
$ brew install python
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
$ sudo pip install --upgrade $TF_BINARY_URL
```

I also needed to run this: 

```
$ rehash
```

[This](http://python-guide-pt-br.readthedocs.io/en/latest/writing/structure/) is a useful guide I used for structuring my codebase.

## Usage 

Execute the code running: 

```
python tensorflow-test.py
```

I would suggest deactivating warnings here: 

```
export TF_CPP_MIN_LOG_LEVEL=2
```

## Resources 

Visual Information Theory - [http://colah.github.io/posts/2015-09-Visual-Information/](http://colah.github.io/posts/2015-09-Visual-Information/)