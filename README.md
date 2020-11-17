# The Effect of Cannabis on the Brain: A Machine Learning Approach to MRI Analysis

Dataset downloaded from https://openneuro.org/datasets/ds000174/versions/1.0.1 and is in the classic BIDS format.
  - 42 Patients (20 Cannabis Users, 22 Controls) each with 2 MRIs (one at a baseline time, and one at a 3-year follow-up)
  - Each MRI has an assigned CUDIT score (higher CUDIT = heavier cannabis consumption) in participants.tsv
  - MRI Images in mri_data folder

Environment Conditions
  1. pip install nilearn
  2. pip install monai
  3. pip install torch
  4. pip install tensorflow
  5. pip install keras
  6. python 3.6.1
 
 Experiments: 
  - Binary Classification of MRI images as "Heavy Cannabis User" or "Control"
  - CUDIT score prediction (higher CUDIT -> more cannabis use)
  - Decoding Voxels of High Predictive Value
 
 How to Run? 
  - in "main.py" there are several hyperparameters denoting which task to run, and how to run it. After configuring, in the command line execute: python main.py
 
 "How can this be adapted to my own project?"
  - Inside the 'mri_data' folder, all the subject data is there and formatted by the BIDS format. This is a consistent format accross many OpenNeuroCV datasets. In "File_Structure.py", there is code for extracting all of the MRI filenames including their 'location paths' and labelling them based on the "participants.tsv" file. 
  - As long as you can create an array X = all filenames including the paths to those files, and y = the labels of these filenames, they can be fed into the "evaluate" function in "run_tests". 
  - You may also want to adjust the self.shape attribute in run_tests to reflect the nature of your dataset. 

Results:

Control vs. Heavy Cannabis User

CUDIT Score

Decoding

Contact Information: 
Aaron: sosaar@stanford.edu
Vivian: vzhu04@stanford.edu
Feel free to reach out!


