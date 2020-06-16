import os
import argparse
import numpy as np
import pandas as pd
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from hw2 import rearrange_dataset
from hw2 import CNN

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()


batch_size = 1


# Image Preprocessing
#The following function gets the path of the all pictures together and it splits
#to the two different class labels'


rearrange_dataset(args.input_folder)
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
])
files=os.listdir(args.input_folder+'/0')+os.listdir(args.input_folder+'/1')
dataset = dsets.ImageFolder(args.input_folder, transform=transform)
cnn=CNN()
cnn.load_state_dict(torch.load('cnn.pkl'))
cnn.eval()
loader=torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#####
# TODO - your prediction code here

y_pred=[]
for images, labels in loader:
    outputs = cnn(images)
    outputs=torch.log(outputs)
    _, predicted = torch.max(outputs.data, 1)
    y_pred.append(predicted)
prediction_df = pd.DataFrame(zip(files, y_pred), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Example - Calculating F1 Score using sklrean.metrics.f1_score
from sklearn.metrics import f1_score
y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

print("F1 Score is: {:.2f}".format(f1))


