# Deaf Translator (Machine Learning Part) Road map 

    👉 [1] to deal with folder of training Images <use python os,shutil>
            🔗 https://www.geeksforgeeks.org/os-module-python-examples/
            🔗 https://docs.python.org/3/library/shutil.html

## 

    👉 [2] to display images in run mode <use seaborn>
            🔗 https://seaborn.pydata.org/

## 

    👉 [3] to prepare Images to training <use torchvision transform>
            🔗 https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html

## 

    👉 [4] to convert the image folder to dataset format <use torchvision dataset>
            🔗 https://pytorch.org/vision/stable/datasets.html
        
## 

    👉 [5] to split dataset to training and testing <use torch random_split>
            🔗 https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

## 

    👉 [6] to combine a dataset(train & test) and a sampler <use torch DataLoader>
            🔗 https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        
## 

    👉 [7] to show image in grid format <use torchvision make_grid>
            🔗 https://pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html
        
## 

    👉 [8] before creating model you should using GPU (CUDA) device to create a model
            🔗 https://pytorch.org/docs/stable/cuda.html
        
## 

    👉 [9] to normalized the data to make model run better <use torchvision transforms>
            🔗 https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
        
## 

    👉 [10] to create the model we will use the (resnet50) model 
            🔗 https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
            🔗 https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            🔗 https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            🔗 https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
            🔗 https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            🔗 https://pytorch.org/docs/stable/generated/torch.no_grad.html

## 

    👉 [11] to save the model after training <use torch.save>
            🔗 https://pytorch.org/docs/stable/generated/torch.save.html

## 

    👉 [12] to test the model using Images
            🔗 https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
            🔗 normalize image go to step number [9]
            🔗 https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
            🔗 https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
