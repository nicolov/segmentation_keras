FROM bvlc/caffe:cpu

ENV MODEL_NAME vgg_conv

RUN pip install tensorflow

CMD cd caffe-tensorflow && \
    ./convert.py  \
    --caffemodel ../$MODEL_NAME.caffemodel \
    --data-output-path ../converted/$MODEL_NAME.npy \
    --code-output-path ../converted/${MODEL_NAME}.py \
    ../dilation8_pascal_voc_deploy.prototxt
