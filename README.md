# neural-combination-of-HCTR
The realization of neural combination of handwritten Chinese text recognition, the model is mainly adapted from fairseq

## Requirement

python 3

pytorch 1.0 or higher

## Usage
1. Clone the repo.
```
   git clone  https://github.com/yudmoe/neural-combination-of-HCTR
```
2. Using ready-made data

Download the prepared data

[semantic_data](https://pan.baidu.com/s/1euRCsvhbt65QEugCjMG_4w)   password：esgg

[nosemantic_data](https://pan.baidu.com/s/1EtFjEIoyfXH8BTRoJDP16g)   password：rb62

[semantic_data_withLM](https://pan.baidu.com/s/1PQarML-7mWFKGwDy_7ekcw)   password：r69c

These folders include over-segmentation,CRNN and attention-based Chinese handwritten string recognition results

3. Run model

When training, make sure that the path of the training and test data sets in the training parameters is valid

```
   python threeinput_training.py --PATH1 /trainingdata1 --PATH2 /trainingdata2 --PATH3 /trainingdata3 --testpath1 /testdata1 --testpath2 /testdata2 --testpath3 /testdata3
```

If the training is interrupted, make sure that the parameters *encoder_a_path*, *encoder_b_path*, *encoder_c_path*, *decoder_path* correspond to their respective encoders when continuing the training

