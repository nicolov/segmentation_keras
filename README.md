Keras implementation of DilatedNet
==================================

A Keras implementation of semantic segmentation according to *Multi-Scale Context Aggregation by Dilated Convolutions (2016)* using the pretrained weights by the [authors'](https://github.com/fyu/dilation).

I have [instructions](TODO) to convert the weights from Caffe to Keras/TF using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow).

Requires Python2.7 due to OpenCV's Python wrapper.

Running
-------

```
pip2 install -r requirements.txt
# Follow TF instructions to install TensorFlow for your platform
python predict.py
```

<hr>

*Fisher Yu and Vladlen Koltun, Multi-Scale Context Aggregation by Dilated Convolutions, 2016*