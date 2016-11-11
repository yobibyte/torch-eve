# torch-eve
An attempt to replicate https://arxiv.org/abs/1611.01505v1 results. Work in progress...

<img src="http://newsinphoto.ru/wp-content/uploads/2011/05/95.jpg" width="400"/>

Right now you can just test it on MNIST. CIFAR evaluation is in progress.

## How to use

### MNIST

```bash                                                                         
  th mnist-example.lua -f                                                     
```

## References

* Original [paper](https://arxiv.org/abs/1611.01505v1).
* The initial code was taken from [optim.adam](https://github.com/torch/optim) and modified to eve.
* The test code is taken from [optim](https://github.com/torch/optim) test folder.
