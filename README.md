# TCAV with Diabetic Retinopathy 

## Model Training

### Files in this repository and what they do: 

| File  | Purpose | Run locally or on Compute Canada? | 
|----------|----------|----------|
| DR_5class_inception_finetune_adam_nopreprocess.py | Transfer learning on EyePACS DR data with InceptionV3  | Compute Canada |
| DR_evaluateModel.py | Evaluate classification metrics on a trained model and save out the y_true and y_pred arrays  | Compute Canada |
| DR_TrueFalseClassification.py  | Generate list of files that were correctly or incorrectly classified by model based on y arrays  | Local |
| Saliency_DR_CC.py | Generate saliency or gradcam maps of DR images | Local |
| TCAV_DR_CC.py | Run TCAV experiment with custom model wrapper and save functions  | Compute Canada |
| plot_tcav_scores.py | Plot TCAV scores for each class, concept, and layer with CSV file saved by TCAV code | Local |


### File structure required for image data generator: 
```
data
├── test_eyepacs
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   └── 4
├── train_eyepacs
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   └── 4
└── val_eyepacs
    ├── 0
    ├── 1
    ├── 2
    ├── 3
    └── 4
```


## TCAV 

Note: `utils_saveplot.py` under my forked `estanley16/tcav` repository contains my modifcations which saves plots instead of just diplaying them, as well as CSV files containing concept, layer, TCAV scores, p values, etc. for that experiment

### Example file structure to run TCAV: 

```
TCAV_cropped_images
#folders with images for each class that will be used to find the TCAV score 
├── class0 
├── class1
├── class2
├── class3
├── class4
#text file with the class labels
├── DR_5class_softmax_labels.txt 
├── hardexudate_cropped #folders containing concept annotations
├── softexudate_cropped
├── hemorrhage_cropped
├── laser_cropped
├── microaneursym_cropped
├── tortuous_cropped
#folder where the model .h5 files are stored
├── models
#folders with random counterexamples, # folders = # replicates for finding that CAV
├── random500_0 
├── random500_1
├── random500_10
├── random500_11
├── random500_12
├── random500_13
├── random500_14
├── random500_15
├── random500_16
├── random500_17
├── random500_18
├── random500_19
├── random500_2
├── random500_20
├── random500_3
├── random500_4
├── random500_5
├── random500_6
├── random500_7
├── random500_8
├── random500_9
#folders where CAVs and activations are dumped with each experiement 
├── tcav_exp1_cropped_model4mixed4_class0 
    ├── activations
    └── cavs
├── tcav_exp1_cropped_model4mixed4_class1
├── tcav_exp1_cropped_model4mixed4_class2
├── tcav_exp1_cropped_model4mixed4_class3
├── tcav_exp1_cropped_model4mixed4_class4
```


