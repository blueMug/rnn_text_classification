Implementing  RNN Text Classification in MXNet
==============================================

A while ago, I implemented a CNN text classification model using MXNet, which can be found [here](https://github.com/blueMug/cnn_text_classification).

This time, I try to implement it in RNN (with attention).

## Data
#### training and validation data
two txt file, the format of each line is: \<label> sentence.

- \<pos> This is the best movie about troubled teens since 1998's whatever.
- \<neg> This 10th film in the series looks and feels tired.

#### config data
one label a line, the number of labels is equals to total classes.
- pos
- neg

#### inference data
one sentence a line, without \<label>

#### inference data with evaluation
the format of each line is: \<label> sentence, like validation file

The data is recommended to be tokenized or segmented(Chinese).

## Quick start
``python rnn_model.py --train path/to/train.data --validate /path/to/validate.data --config /path/to/config``

``python inference.py --test python/to/inference.data --config /path/to/config --checkpoint 1``

``python inference.py --test python/to/inference-evaluation.data --config /path/to/config --checkpoint 1 --evaluation``

## References
- [Implementing CNN Text Classification in MXNet](https://github.com/blueMug/cnn_text_classification)
