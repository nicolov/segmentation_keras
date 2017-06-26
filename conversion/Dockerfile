FROM bvlc/caffe:cpu

ENV MODEL_NAME dilation8_pascal_voc

RUN pip install tensorflow

CMD cd caffe-tensorflow && \
    ./convert.py  \
    --caffemodel "../${MODEL_NAME}.caffemodel" \
    --data-output-path "../converted/${MODEL_NAME}.npy" \
    --code-output-path "../converted/${MODEL_NAME}.py" \
    "../${MODEL_NAME}_deploy.prototxt"
