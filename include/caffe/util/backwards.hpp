//
// Created by Troy Liu on 2019-05-03.
//

#ifndef CAFFE_BACKWARDS_HPP
#define CAFFE_BACKWARDS_HPP

#define NO_BACKWARD(propagate_down) {for (auto down : propagate_down) {if (down) NOT_IMPLEMENTED;}}

#endif //CAFFE_BACKWARDS_HPP
