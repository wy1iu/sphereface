from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

# The initialization used
xavier_constant = dict(type='xavier')
gaussian_constant = dict(type='gaussian', std=0.01)


class Spherenet(object):
    def __init__(self):
        pass

    def conv_prelu(self, bottom, num_out, kernel_size=3, stride=1, pad=1, is_bias=False, 
        wf=gaussian_constant):
    
        if is_bias:
            learn_param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]
        else:
            learn_param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)]
        conv = L.Convolution(bottom,
                             kernel_size=kernel_size,
                             stride=stride,
                             num_output=num_out,
                             pad=pad,
                             param=learn_param,
                             weight_filler=wf,
                             bias_filler=dict(type='constant', value=0))
        prelu = L.PReLU(conv, in_place=True)
        return prelu

    def add_block(self, bottom, num_output):
        layer1 = self.conv_prelu(bottom, num_output)
        layer2 = self.conv_prelu(layer1, num_output)
        output = L.Eltwise(bottom, layer2, eltwise_param=dict(operation=1))
        return output

    def build_convolution(self, bottom, num_output, num_block):
        model = self.conv_prelu(bottom, num_output, 3, 2, 1, True, xavier_constant)
        for i in range(num_block):
            model = self.add_block(model, num_output)
        return model

    def make_net(self, data_file, block_nums,  batch_size=256, feature_dim=512, class_num=10572):
        assert len(block_nums) == 4, print('cause there four convolutions')
        data, label = L.ImageData(image_data_param=dict(source=data_file,
                                                   batch_size=batch_size,
                                                   shuffle=True),

                             transform_param=dict(mean_value=[127.5, 127.5, 127.5],
                                                  scale=0.0078125,
                                                  mirror=True),
                             name='data',
                             ntop=2)

        conv1 = self.build_convolution(data, 64, block_nums[0])
        conv2 = self.build_convolution(conv1, 128, block_nums[1])
        conv3 = self.build_convolution(conv2, 256, block_nums[2])
        conv4 = self.build_convolution(conv3, 512, block_nums[3])

        fc5 = L.InnerProduct(conv4,
                             num_output=feature_dim,
                             bias_term=True,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'),
                             bias_filler=dict(type='constant'),
                             name='fc5')

        fc6 = L.MarginInnerProduct(fc5,
                                   label,
                                   num_output=class_num,
                                   type=3,
                                   param=dict(lr_mult=1, decay_mult=1),
                                   weight_filler=dict(type='xavier'),
                                   base=1000,
                                   gamma=0.12,
                                   power=1,
                                   lambda_min=5,
                                   iteration=0)

        loss = L.SoftmaxWithLoss(fc6, label)
        return to_proto(loss)


def make_net():
    # make spherenet20
    model = Spherenet()
    with open('spherenet_model20.prototxt', 'w') as f:
        block_nums = [1, 2, 4, 1]
        print(str(model.make_net('data/CASIA-WebFace-112X96.txt', block_nums)), file=f)

    # make spherenet32
    with open('spherenet_model36.prototxt', 'w') as f:
        block_nums = [2, 4, 8, 2]
        print(str(model.make_net('data/CASIA-WebFace-112X96.txt', block_nums)), file=f)

    # make spherenet64
    with open('spherenet_model64.prototxt', 'w') as f:
        block_nums = [3, 8, 16, 3]
        print(str(model.make_net('data/CASIA-WebFace-112X96.txt', block_nums)), file=f)


if __name__ == '__main__':
    make_net()
