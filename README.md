<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

# Souffle

This is a DNN fusion framework that implements our paper *"Think Big, Act Small: Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions"* and is based on TVM v0.8.
In this released version it can fuse ResNext-like parallel branches into one tensor expression (TE).
For example, it can fuse the following subgraph
```
          conv2d   conv2d     conv2d
            |         |          |
            |         |          |
      batch_norm   batch_norm  batch_norm
            |         |          |
            |         |          |
          relu       relu      relu
            \         |         /
             \        |        /
                   concat     
```
into
```
            fused(conv2d+batch_norm+relu)
```
The equivalent TE of the `fused(conv2d+batch_norm+relu)` is 
```Python
def fused_conv3x3_bn_relu(batch, height, width, in_channels, out_channels, kernel_h, kernel_w, num_input):
  input_tensor = te.placeholder((batch, num_input, height, width, in_channels), "float32", name="input_tensor")
  weight_tensor = te.placeholder((num_input, kernel_h, kernel_w, in_channels, out_channels), name="weight_tensor")
  bnw1 = te.placeholder((batch, num_input, out_channels), name="bnw1")
  bnw2 = te.placeholder((batch, num_input, out_channels), name="bnw2")

  padded_input_tensor = te.compute((batch, num_input, height+2, width+2, in_channels), \
    lambda b, n, h, w, ic: te.if_then_else(te.all(h>0, h<height-1, w>0, w<width-1), input_tensor[b, n, h-1, w-1, ic], 0))
  rk = te.reduce_axis((0, in_channels), name="rk")
  rx = te.reduce_axis((0, kernel_h), name="rx")
  ry = te.reduce_axis((0, kernel_w), name="ry")
  conv_output = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, oc: te.sum(padded_input_tensor[b, n, h+rx, w+rx, rk] * weight_tensor[n, rx, ry, rk, oc], axis=[rk, rx, ry]))
  bn_multiply = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: conv_output[b, n, h, w, o] * bnw1[b, n, o])
  bn_add = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: bn_multiply[b, n, h, w, o] + bnw2[b, n, o])
  output = te.compute((batch, num_input, height, width, out_channels),\
    lambda b, n, h, w, o: tir.max(bn_add[b, n, h, w, o], 0))

  return [input_tensor, weight_tensor, bnw1, bnw2, output]
```
The variable `num_input` represents the number of branches. And the computation of `bn_multiply` `bn_add`, and `output` are all scheduled by `compute_inline`.

This project is still in active development.

## How to apply

First check out TVM from github:
```shell
git clone https://github.com/apache/tvm.git
git checkout v0.8.0
```

Clone this repo:

```shell
https://github.com/summerspringwei/souffle-patch.git
```

Copy the patch file to the root path of tvm and apply:
```shell
cd souffle-patch
cp souffle.patch path/to/tvm
cd path/to/tvm
git apply souffle.patch
```

## Build and run

First, follow the [instructions](https://tvm.apache.org/docs/v0.8.0/install/from_source.html#install-from-source) on the TVM official website to build and install this project.
Then run the following test the verify the functionality of the project.
```shell
cd tests/python/relay/
python3 test_souffle_fusion.py
```

# Apache TVM
Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.

Getting Started
---------------
Check out the [TVM Documentation](https://tvm.apache.org/docs/) site for installation instructions, tutorials, examples, and more.
The [Getting Started with TVM](https://tvm.apache.org/docs/tutorial/introduction.html) tutorial is a great
place to start.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Check out the [Contributor Guide](https://tvm.apache.org/docs/contribute/).

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
  originates from Halide. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
