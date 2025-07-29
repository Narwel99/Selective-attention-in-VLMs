# Selective-attention-in-VLMs
Analysis of behavioral selective attention in LLAVA-NEXT-VIDEO-8b.  
Does filtering out bakcground noise to focus on relevant object in a scene enhance VLMs performance in understanding Temporality and objects-actions causality?

## Project content
This Project contains:  
Dataset:
  - SSV2 dataset zip files for Segmentation
  - VideoTest (the original video of the 100 videos I've segmented, used for Model inference)
  - Ground_truth.json and prompt.json files (I have hand-crafted 5 prompts and 5 answers as ground truth for 30 videos, these files are used for Model inference)
  - labels.json (file that containt the labels from SSV2 only for the 100 segmented videos, used for testing)
Outputs:
  - masks zip files (used for model inference Crop-and-mask mode)
  - Video (this fodler show some examples of segmented videos, it show the recosntructed video with the bounding box, Ids, and mask (color change on masked region)
  - attention maps (this folder show some examples on how the model 'sees' in different modes)
  - Tables (my results of model inference)

Codes:
  - ManualSegmentation code
  - Model_2modes
  - Requirement.txt for each code
  - ReadME.

This project has two step.  
First : Segmentation process  
Second : Models inference

## INSTALLATION AND RUNNING INSTRUCTIONS:
  ### Segmentation process
- Download the Project files
- create new environment with specific python version for Segmentation codes (makes sure you have conda installed)
  ```
  conda create -n Segm python=3.11.12
  ```
  activate environment
   ```
  conda activate Segm 
  ```
  -Download all necessary libraries, you can use the requirements.text files  
  There is two of them, for Segmentation use requirementsS.txt
   ```
  pip install -r /path/to/requirementsS.txt
  ```
  -open ManualSegmentation notebook
  -Select Appropriate Kernel (here Segm)
  -run the cells

  ### Model inference
  - create another environment with specific python version for the Model
  ```
  conda create -n Model python=3.10.18
  ```
  activate environment
   ```
  conda activate Model 
  ```
  -Download all necessary libraries, you can use the requirements.text files
  Here us requirementsM.txt
   ```
  pip install -r /path/to/requirementsM.txt
  ```
  -open Model_2modes notebook
  -Select Appropriate Kernel (here Model)

## CODES EXPLANATION
### Segmentation process
This codes allows to visualize videos from the SSV2 dataset.  
It provide a custom interface tho guide the segmentation processus  
The interface allow the user to:
- visualize video frames of the current video by skipping each 10th frames or selecting a specific frame.
- add or remove points in the current frames (points will create a bounding box to the object, that will be display right after the selection)
- skip the current video or jump to a specific video (by its name)
- save the current selection, automatically segment the videos and gather results then switch to next video.
- stop the processus.

Everytimes a segmentation is performed the code saves:
- the frames (for visualization)
- the frames with the bouding boxes (for visualization)
- the frames with the masks (for visualization)
- the segmentation masks (.npy files used for Model inference process)
- the reconstructied video with the bouding boxes , IDs and masks
- the reconstructied video with masks only (different colors on objects for visualization)
- the tracking results for each objects accross frames
- the selections made (coordinate points to select an object for segmentation)

In the outputs directory, for a matter of size you can only see a few examples of the reconstructed videos for both cases.  
And the masks file for the 100 videos I have segmented. (zip file), which is needed to run the next step

### Model inference
This codes runs the model in two modes.  
The original mode is the LLaVa-Next-Video-b model as is.  
The crop-and-mask model uses the msegmentation masks and applies it to the frame of the video, crop and mask the frame based on the region given by the mask to create 'a focus' on the frame, zeroing out everything outside the region (the aim is to avoid background bias).  

The custom interface allows to choose two options:  
option 1 (single video analysis) analyse and provide results for only one video :
- write the name of the video for analysis
- write the prompts
  
  Results for each modes :
  - Question category (categorical / open-ended)
  - Model answers for each questions
  - Accuracy for each categorical questions
  - F1 Score for each categorical questions
  - RougeL score for each open-ended questions
  - Semantic similarity for each open-ended questions
  - Total scores of metrics for the mode

option 2 (all videos analysis) analyse and provide results for all the videos  
This mode look at the prompts.json file and run the two modes for all the videos.  
If the human notation is enables, during process the code will provides the results of each question as option 1 and ask for the notation of each questions.  
This is time consuming depending on the amount of video and prompt (here 297 answers) so i disable it.  
At the end this option return the result in two CSV files.  
  Description_table : 
    - Name of the video
    - Current Mode
    - Questions
    - Questions category
    - Accuracy 
    - F1 score 
    - RougeL score
    - Semantic similarity
    - Model Answers
    - Ground truth
    - Human notation
  
  Summary_table :
  For each mode the total score (mean of metrics)



