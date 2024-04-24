# ViTFL: Vision Transformer in Federated Learning for Classification Tasks
> This is the official implementation of ViTFL

## Preparation

1.Download the datasets and use datasplit.py to allocate the data to each client.


2.Move the processed datasets to ./data/DATASETS_HERE/client_X

## Installation
1.Requirments:
keras                        2.6.0

matplotlib                   3.7.4

numpy                        1.24.4

pandas                       2.0.3

pytorch-gradcam              0.2.1

scipy                        1.10.1

tensorflow                   2.6.0

tensorflow-estimator         2.10.0

tensorflow-gpu               2.6.0

tensorflow-intel             2.13.0

tensorflow-io-gcs-filesystem 0.31.0

torch                        1.8.1+cu111

torchaudio                   0.8.1

torchcam                     0.4.0

torchsummary                 1.5.1

torchvision                  0.9.1+cu111

Or you can use the command below to install all requirments easily:

```sh
pip install -r requirements.txt
```


Use the command below to run the model:

```sh
python model.py
```

To test the model:

```sh
python test_acc.py
```




