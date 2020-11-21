# Recognizing Diseased Coffee Leaves Using Deep Learning


In this project, given a set of images of coffee leaves, this project will explore deep learning algorithms (both fully connected and convolution
neural networks) that output the correct labels for the conditions of the coffee leaves. 

The six conditions are 
* Healthy (H)
* Rust Level 1 (RL1)
* Rust Level 2 (RL2)
* Rust Level 3 (RL3)
* Rust Level 4 (RL4)
* Red Spider Mites (RSM)

For this project, we explore three different tasks:
1) Given the full dataset, classify them into the 6 categories mentioned above.
2) Given the full dataset, classify them into 3 categories (H, RL, RSM).
3) Given the images from the healthy and rust level categories only, classify them into 5 categories (H, RL1, RL2, RL3, RL4) 
using a regression-based approach.

#### Datasets
We make use of the robusta coffee dataset (RoCoLe), which contains annotations and images and can be downloaded [here](https://data.mendeley.com/datasets/c5yvn32dzg/2).

### Setting the Virtual Environment and Installing Requirements
Requirements:
We recommend using python3 and a virtual env. Run the follow commands:
```sh
$ virtualenv -p python3 .env
$ source .env/bin/activate
$ pip install -r code/requirements.txt
```

### Processing the Dataset
After downloading the annotations and images, they should be placed inside the ```cs23-project-coffee-leaf-disease``` directory as follows
* ```cs23-project-coffee-leaf-disease/Annotations/{annotation files}```
* ```cs23-project-coffee-leaf-disease/Photos/{.jpg files}```

To process the images for Task 1 above, run the following command:
```sh
python dataProcessing.py
```

To process the images for Task 2 above, run the following command:
```sh
python dataProcessingForThreeClasses.py
```

To process the images for Task 3 above, run the following command:
```sh
python dataProcessingForRegressionTask.py
```

### Training models
To train models, first create a ```params.json``` file inside the ```experiments/{A}/{B}``` directory, where
* {A} is either ```six_classes```, ```three_classes```, or ```regression```
* {B} is the descriptive name for the experiment model

Then for Task 1 and 2, we use ```train.py``` to train a model. Simply run
```sh
python train.py --model_dir experiments/{A}/{B}
```

For Task 3, we use the ```trainRegression.py``` file. Simply run
```sh
python train.py --model_dir experiments/regression/{B}
```

### Evaluate on a Saved Model
To evaluate on a saved model, run
```sh
python calculateF1Metrics --model_dir experiments/{A}/{B}
```
By default, it will evaluate on only the training and validation set. 
* To evaluate on the test set, the 
```--testSet True``` flag must be added. 
* To not evaluate on the training and validation set, we can set the 
```--trainAndVal False```.
* To evaluate only on the test set, we can set both flags ```--trainAndVal False --testSet True```


# Acknowlegements
We would like to thank Surag Nair, Olivier Moindrot and Guillaume Genthial for the PyTorch Vision Deep Learning starter 
code, which can be foudn [here](https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision).