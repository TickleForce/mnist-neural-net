This is a Rust implementation of a simple neural network for recognizing handwritten digits. It is trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

This is a self-contained project without any dependencies besides the Rust standard library. It was built as a learning exercise, without tremendous thought for efficiency. In particular, the number of memory allocations could be cut down significantly to increase performance.

Much thanks to Michael Nielsen for his book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html). This project is primarily an implementation of the concepts and examples taught there.