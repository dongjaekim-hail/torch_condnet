# torch_condnet
torch version of condnet by e bengio


# requirements

## CUDA 
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
## CPU
```commandline
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```


# save conda env to yaml

```commandline
conda env export > environment.yml
```

# install yaml from environment.yml

```commandline
conda env create -f environment.yml
```
