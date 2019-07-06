
# Creating and deploying a custom image classifier


```
import fastai
fastai.__version__
```




    '1.0.54'




```
# Mount Google drive
from google.colab import drive
drive.mount('/content/drive')
```

## 1. Libraries



```
from fastai import *
from fastai.vision import *
```

## 2. To download URLs of Google Images


Copy paste this in the browser console (F12) to download the URLs of all the searched Google images.

```javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```




```
classes = ['greenwing', 'hahn', 'hyacinth', 'scarlet'] # define your classes here
files = ['urls_'+x+'.txt' for x in classes]
files
```




    ['urls_greenwing.txt',
     'urls_hahn.txt',
     'urls_hyacinth.txt',
     'urls_scarlet.txt']




```
path = 'drive/My Drive/fast.ai/macaws/' # Specify path
path = Path(path)
path.ls()
```




    [PosixPath('drive/My Drive/fast.ai/macaws/urls_greenwing.txt'),
     PosixPath('drive/My Drive/fast.ai/macaws/urls_hahn.txt'),
     PosixPath('drive/My Drive/fast.ai/macaws/urls_hyacinth.txt'),
     PosixPath('drive/My Drive/fast.ai/macaws/urls_scarlet.txt'),
     PosixPath('drive/My Drive/fast.ai/macaws/greenwing'),
     PosixPath('drive/My Drive/fast.ai/macaws/hahn'),
     PosixPath('drive/My Drive/fast.ai/macaws/hyacinth'),
     PosixPath('drive/My Drive/fast.ai/macaws/scarlet'),
     PosixPath('drive/My Drive/fast.ai/macaws/models'),
     PosixPath('drive/My Drive/fast.ai/macaws/cleaned.csv')]



### Download Images


```
for idx, c in enumerate(classes):

    file = files[idx]
    download_images(path/file, path/c, max_pics=200)
```

### Verify and delete images


```
for c in classes:
    print('class :', c)
    verify_images(path/c, delete=True, max_workers=8)
```

### View data


```
imagenet_stats
```




    ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])




```
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
                                 ds_tfms=get_transforms(), size=224,
                                 num_workers=4).normalize(imagenet_stats)
```


```
data.classes
```




    ['greenwing', 'hahn', 'hyacinth', 'scarlet']




```
data.show_batch(rows=5, fig_size=(5,5))
```


![png](/images/1.png)



```
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```




    (['greenwing', 'hahn', 'hyacinth', 'scarlet'], 4, 278, 69)



## Train Model


```
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
```


```
# Print out the model/architecture
learn.summary()
```




    Sequential
    ======================================================================
    Layer (type)         Output Shape         Param #    Trainable
    ======================================================================
    Conv2d               [64, 112, 112]       9,408      False     
    ______________________________________________________________________
    BatchNorm2d          [64, 112, 112]       128        True      
    ______________________________________________________________________
    ReLU                 [64, 112, 112]       0          False     
    ______________________________________________________________________
    MaxPool2d            [64, 56, 56]         0          False     
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    ReLU                 [64, 56, 56]         0          False     
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    ReLU                 [64, 56, 56]         0          False     
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    ReLU                 [64, 56, 56]         0          False     
    ______________________________________________________________________
    Conv2d               [64, 56, 56]         36,864     False     
    ______________________________________________________________________
    BatchNorm2d          [64, 56, 56]         128        True      
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        73,728     False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    ReLU                 [128, 28, 28]        0          False     
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        8,192      False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    ReLU                 [128, 28, 28]        0          False     
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    ReLU                 [128, 28, 28]        0          False     
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    ReLU                 [128, 28, 28]        0          False     
    ______________________________________________________________________
    Conv2d               [128, 28, 28]        147,456    False     
    ______________________________________________________________________
    BatchNorm2d          [128, 28, 28]        256        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        294,912    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        32,768     False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    ReLU                 [256, 14, 14]        0          False     
    ______________________________________________________________________
    Conv2d               [256, 14, 14]        589,824    False     
    ______________________________________________________________________
    BatchNorm2d          [256, 14, 14]        512        True      
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          1,179,648  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    ReLU                 [512, 7, 7]          0          False     
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          2,359,296  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          131,072    False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          2,359,296  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    ReLU                 [512, 7, 7]          0          False     
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          2,359,296  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          2,359,296  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    ReLU                 [512, 7, 7]          0          False     
    ______________________________________________________________________
    Conv2d               [512, 7, 7]          2,359,296  False     
    ______________________________________________________________________
    BatchNorm2d          [512, 7, 7]          1,024      True      
    ______________________________________________________________________
    AdaptiveAvgPool2d    [512, 1, 1]          0          False     
    ______________________________________________________________________
    AdaptiveMaxPool2d    [512, 1, 1]          0          False     
    ______________________________________________________________________
    Flatten              [1024]               0          False     
    ______________________________________________________________________
    BatchNorm1d          [1024]               2,048      True      
    ______________________________________________________________________
    Dropout              [1024]               0          False     
    ______________________________________________________________________
    Linear               [512]                524,800    True      
    ______________________________________________________________________
    ReLU                 [512]                0          False     
    ______________________________________________________________________
    BatchNorm1d          [512]                1,024      True      
    ______________________________________________________________________
    Dropout              [512]                0          False     
    ______________________________________________________________________
    Linear               [4]                  2,052      True      
    ______________________________________________________________________

    Total params: 21,814,596
    Total trainable params: 546,948
    Total non-trainable params: 21,267,648
    Optimized with 'torch.optim.adam.Adam', betas=(0.9, 0.99)
    Using true weight decay as discussed in https://www.fast.ai/2018/07/02/adam-weight-decay/
    Loss function : FlattenedLoss
    ======================================================================
    Callbacks functions applied




```
learn.model
```




    Sequential(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (4): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (5): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): Sequential(
        (0): AdaptiveConcatPool2d(
          (ap): AdaptiveAvgPool2d(output_size=1)
          (mp): AdaptiveMaxPool2d(output_size=1)
        )
        (1): Flatten()
        (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout(p=0.25)
        (4): Linear(in_features=1024, out_features=512, bias=True)
        (5): ReLU(inplace)
        (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout(p=0.5)
        (8): Linear(in_features=512, out_features=4, bias=True)
      )
    )




```
learn.fit_one_cycle(4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.791311</td>
      <td>1.038643</td>
      <td>0.463768</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.212988</td>
      <td>0.429476</td>
      <td>0.173913</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.923938</td>
      <td>0.368049</td>
      <td>0.159420</td>
      <td>00:12</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.753958</td>
      <td>0.356485</td>
      <td>0.130435</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>



```
learn.save('stage-1')
```


```
learn.unfreeze()
```


```
learn.lr_find()
```





    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.


### Find an approximate mid point in the steepest downward  slope


```
learn.recorder.plot(suggestion=True)
```

    Min numerical gradient: 2.29E-04
    Min loss divided by 10: 2.51E-04



![png](/images/2.png)



```
min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr
```




    0.00022908676527677726




```
learn.load('stage-1')
learn.unfreeze()
```


```
learn.fit_one_cycle(2, max_lr=min_grad_lr)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.264075</td>
      <td>0.338774</td>
      <td>0.101449</td>
      <td>00:13</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.221863</td>
      <td>0.423632</td>
      <td>0.115942</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>



```
learn.save('stage-2')
```

## Interpretation


```
learn.load('stage-2')
```



```
intrp = ClassificationInterpretation.from_learner(learn)
intrp.plot_confusion_matrix()
```


![png](/images/3.png)


Around 6% error rate


```
intrp.most_confused()
```




    [('greenwing', 'scarlet', 3), ('greenwing', 'hahn', 1)]




```
intrp.plot_top_losses(4)
```


![png](/images/4.png)


## Cleaning Noisy Images


```
from fastai.widgets import ImageCleaner
from fastai import *
```


```
losses, idxs = intrp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]
```


```
top_loss_paths
```




    ImageList (69 items)
    Image (3, 300, 400),Image (3, 956, 1300),Image (3, 369, 458),Image (3, 1280, 1159),Image (3, 1080, 1916)
    Path: drive/My Drive/fast.ai/macaws




```
db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )
```


```
db
```




    ImageDataBunch;

    Train: LabelList (347 items)
    x: ImageList
    Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224),Image (3, 224, 224)
    y: CategoryList
    hahn,hahn,hahn,hahn,hahn
    Path: drive/My Drive/fast.ai/macaws;

    Valid: LabelList (0 items)
    x: ImageList

    y: CategoryList

    Path: drive/My Drive/fast.ai/macaws;

    Test: None




```
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
learn_cln.load('stage-2');
```


```
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
```


```
# Jupyter Widgets (ipywidgets) does not work on Colab - try local jupyter notebook!
# ImageCleaner(ds, idxs, path)
```


```
df = pd.read_csv(path/'cleaned.csv', header='infer')
```

## Deploying to Production


```
data.classes
```




    ['greenwing', 'hahn', 'hyacinth', 'scarlet']



For inference we can use CPU


```
path
```




    PosixPath('drive/My Drive/fast.ai/macaws')




```
img = open_image(path/'hahn'/'00000014.jpg')
img
```




![png](/images/5.png)




```
# Prepare ImageDataBunch with the same transformations - ideally run this once
# when the app loads
classes = ['greenwing', 'hahn', 'hyacinth', 'scarlet']
data1 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(),
                                           size=224).normalize(imagenet_stats)
learn1 = cnn_learner(data1, models.resnet34)
learn1.load('stage-2')
```


```
pred_class, pred_idx, outputs = learn1.predict(img)
pred_class
```




    Category hahn



## Export Model


```
learn1.export()
```

This exported trained model can now be  served in the backend  for inference on the web.
