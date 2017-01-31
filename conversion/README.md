# Weight conversion

Pull in the network definition and Caffe weights trained by the authors (example for Pascal VOC):

    wget -O vgg_conv.caffemodel https://umich.box.com/shared/static/obagzzef426d3ty4tihgscr17nxmgvo1.caffemodel
    wget https://raw.githubusercontent.com/fyu/dilation/master/models/dilation8_pascal_voc_deploy.prototxt

Clone caffe-tensorflow in the current directory:

    git clone https://github.com/ethereon/caffe-tensorflow

Build and run the Dockerfile:

    docker run -v $(pwd):/workspace -ti `docker build -f Dockerfile -q .`

Weights and code end up in `converted/`
