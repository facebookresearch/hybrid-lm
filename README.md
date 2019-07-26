# Hybrid-LM
This is the code to reproduce the experiments in the "Training Hybrid Language Models by Marginalizing over Segmentations" paper.

## Requirements
The code requires:
* A C++ compiler with good C++11 support (e.g. g++ >= 4.8)
* [cmake](https://cmake.org/) — version 3.5.1 or later, make
* [ArrayFile](https://github.com/arrayfire/arrayfire/wiki) (>= 3.6.1) is required
* [flashlight](https://github.com/facebookresearch/flashlight/) is required.

## Building
Please follow the instruction [here](https://fl.readthedocs.io/en/latest/installation.html) to build flashlight as well as this project.

## Running the code
The following command line can be used to reproduce experiments on the MWC datasets:
```
./transformer --data PATH/TO/DATA --bsz 8 --bptt 512 --d_model 512 --d_ff 2048 --n_blocks 12 --n_heads 4 --warmup 8000 --warmup_loss 20 --ngram 4 --threshold 200 --lr 0.025 --clip 0.1 --dropout 0.3 --use_cache --nepoch 30
```

## License
The code is licensed under CC-BY-NC, as found in the LICENSE.txt file.
