# Selective-attention-in-VLMs
***Bridging Biological Cognition and Multimodal AI***

This repository contains the framework and experimental pipeline developed for my Master's Thesis: **"Analysis of temporal and causal understanding in VLMs in comparison with the biological brain."**

## Research Objective
Does simulating biological behavioral selective attention enhance the performance of Vision-Language Models (VLMs) in understanding Temporality and Object-Action Causality?

By filtering out background noise to focus strictly on relevant objects (simulating the "What" and "Where" pathways), this project evaluates whether current architectures like LLaVA-NeXT-Video, mPLUG-Owl3, and Video-LLaMA2 can achieve better cognitive alignment with human-like reasoning

## Repository Structure
```text
.
├── datasets/
│   └── VideoTest/             # 100 selected .webm clips from SSV2 dataset
├── outputs/
│   ├── SegmentedVideos/       # Visual proof: Videos with Boxes, IDs, and Mask Overlays
│   ├── attention_maps/        # Comparative visuals: Original vs. Crop-and-Mask (Foveal Focus)
│   ├── Masks.zip/             # Pre-generated .npy masks for rapid inference replication
│   ├── detailed_evaluation/   # Full results (Excel: 296 QA pairs with ROUGE-L, Sim, etc.)
│   └── summary_tables/        # Aggregate performance & Mean Accuracy tables
├── Manual.ipynb               # Custom GUI for SAM-based Segmentation & Tracking
├── Model_LLaVA.ipynb          # Inference & Evaluation pipeline for LLaVA-NeXT-Video
├── Model_mPLUG.ipynb          # Inference & Evaluation pipeline for mPLUG-Owl3
├── Model_VideoLLaMA.ipynb     # Inference & Evaluation pipeline for Video-LLaMA2
├── requirementsS.txt          # Dependencies for Phase 1 (Segmentation/SAM)
└── requirementsM.txt          # Dependencies for Phase 2 (VLM Inference)
```
This project has two step.  
First : Segmentation process  
Second : Models inference and Evaluation

## Installation & Running :
  ### Segmentation Pipeline (Segment_anything_model, SAM)
  ```
  conda create -n Segm python=3.11.12
  conda activate Segm
  pip install -r requirementsS.txt
  ```
 Run Manual.ipynb. Ensure you select the Segm kernel.

  ### VLM inference
  ```
  conda create -n Model python=3.10.18
  conda activate Model
  pip install -r requirementsM.txt
  ```
 Run the specific Model_xxx.ipynb for the model you wish to test. Ensure you create different environments for each models, avoiding dependencies conflicts.

## Framework Explanation
### Phase 1: Biologically-Inspired Preprocessing (Manual.ipynb)
This notebook provides a custom interface to guide the segmentation process:
 - **Interactive Selection:** Add/remove points to define objects of interest.(points will create a bounding box to the object, that will be display right after the selection)
 - **Visualization:** travels accross videos frames or differents videos.
 - **Automated Tracking:** Leverages SAM to track and segment causal agents across all frames.
 - **Output Generation:** Saves .npy masks, bounding boxes, and IDs for the next phase.
  
Everytimes a segmentation is performed the code saves:
- the frames (for visualization)
- the frames with the bouding boxes (for visualization)
- the frames with the masks (for visualization)
- the segmentation masks (.npy files used for Model inference process)
- the reconstructed video with the bouding boxes , IDs and masks
- the reconstructied video with overlays on the objects only
- the tracking results for each objects accross frames
- the selections made (coordinate points to select an object for segmentation)

### Phase 2: Dual-Mode Inference (Model_xxx.ipynb)
The models are evaluated under two conditions:
1. **Original Mode (Baseline):** The VLM processes the video as-is.
2. **Crop-and-Mask:** Based on the generated masks, the video is cropped to isolate the objects of interest and zero-out the background, simulating foveal focus and removing environmental bias.
 
**Inference Options:**
- **Single Video Analysis:** Deep dive into a specific clip with real-time scoring.
- **Batch Analysis:** Automated run over the entire dataset, generating Description_table.csv and Summary_table.csv.

