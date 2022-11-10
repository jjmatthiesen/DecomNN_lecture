# Training CNN with Cat and Dog Images

## Dataset
1. download the "Cat and Dog" dataset from kaggle: 
    > https://www.kaggle.com/datasets/tongpython/cat-and-dog

2. Move the data to the /data/catsDogs folder
3. Remove the recurrent folders. The structure should like like the following 

``` 
data
└───catsDogs
│   └───trainings_set
│       │    └───  cats
│       │    └───  dogs
│   └───test_set
│       │    └───  cats
│       │    └───  dogs

```

4. remove duplicates
   1. The trainings_set/dogs folder contains duplicates. You can find them easily by name (e.g. dog.2000(1).jpg).
   2. You can delete the duplicates by using `rm dog*\(1\).jpg` via the terminal in the folder. 
   After removing both folders (/cats and /dogs) should contain 4000 images.  
5. create a validation set.
   1. We need a validation set. For that we will split the data in /training_set into two parts.
   2. We want to have 80% of the images in the /training_set folder and 20% in a new folder called /validation_set.
   3. If you navigate to the /data folder in the terminal you can use the `train_val_split.sh` script to do so.
   4. After this step your structure should look as follows:
``` 
data
└───catsDogs
│   └───trainings_set
│       │    └───  cats
│       │    └───  dogs
│   └───test_set
│       │    └───  cats
│       │    └───  dogs
│   └───validation_set
│       │    └───  cats
│       │    └───  dogs

```

6. Try to load the images and create a data loader via the first lines in train.py. 
Depending on you root, you might need to change the folder path.