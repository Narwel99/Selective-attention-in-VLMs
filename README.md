# Selective-attention-in-VLMs
***Bridging Biological Cognition and Multimodal AI***  
This repository contains the framework, dataset, and experimental pipeline developed for my Master's Thesis: **"Analysis of temporal and causal understanding in VLMs in comparison with the biological brain."**

## Research Objective
Current Vision-Language Models (VLMs) often struggle with temporal reasoning and causal inference, relying on statistical co-occurrence rather than true understanding. Unlike the biological brain, which uses a behavioral selective attention (filtering out background noise to focus on relevant objects), VLMs process all pixels equally before attention.

**Core research question:** *Does simulating biological behavioral selective attention enhance the performance of Vision-Language Models (VLMs) in understanding Temporality and Object-Action Causality?*  
* **H1:** Implementing behavioural selective attention improves VLM’s performance in temporal and causal reasoning.
* **H0:** Behavioural selective attention has no significant effect on VLM’s performance in temporal and causal reasoning.

## Methodology:
To test this, I engineered a dual-phase pipeline to compare how models react to standard video inputs versus stimuli modified to mimic human selective attention.
### Phase 1: Biologically-Inspired Preprocessing (`Manual.ipynb`)
Using the **Segment Anything Model (SAM)**, this custom interactive tool processes the raw videos to simulate visual focus:
1. **Interactive Selection:** Manually define the causal objects ("Agent" and "Patient") in a frame.
2. **Automated Tracking:** SAM tracks and segments these entities across all frames.
3. **Stimulus Generation:** Generates binary masks (`.npy`) and reconstructed videos where the background is completely zeroed-out.

### Phase 2: Dual-Condition Inference (`Model_xxx.ipynb`)
Three state-of-the-art architectures were evaluated on QA pairs under two conditions:
* **Original Mode (Baseline):** The VLM processes the video as-is (diffuse attention).
* **Crop-and-Mask Mode:** The VLM processes only the segmented objects (forced selective attention).
  
**Evaluation Metrics:** Keyword Accuracy, Semantic Similarity (via SentenceTransformers), Human evaluation and Qualitative Vision Tower Heatmap Analysis.

## Key Findings & Results
The experimental results validate the **Null Hypothesis (H0)**: Enforcing attention to relevant objects is *not* a universal solution. Its effectiveness is strictly **model-dependent** and tied to the underlying architecture.

### 1. Quantitative Performance Across Reasoning Domains
Models were evaluated across different cognitive levels: Object-Action Recognition, Temporal Reasoning, and Causal Reasoning.  
<img width="953" height="562" alt="image" src="https://github.com/user-attachments/assets/cb81e061-ef9d-4be8-bbef-c5a833de86cd" />

* **LLaVA-NeXT-Video (7B) ➔ BENEFITS MOST:** Improves across all reasoning domains. The model becomes more consistent, with a notable decrease in hallucination rates and internal contradictions accross questions. It struggles significantly less with higher-order reasoning tasks once the foveal focus is applied.
* **VideoLLaMA2 (7B) ➔ NEUTRAL BENEFITS:** Shows slight improvements across reasoning domains and is slightly more consistent (internal contradictions are reduced). However, it hallucinates more. Overall, it performs slightly better in higher reasoning tasks but the gains remain modest.
* **mPLUG-Owl3 (7B) ➔ DOES NOT BENEFIT:** Decreases in temporal and causal reasoning performance. Notably, adding more frames (from 8f to 16f) worsens all metrics. It becomes slightly less consistent, hallucination rates increase, and while internal contradictions are reduced, it struggles heavily with higher reasoning tasks.

### 2. Interpretability: Inside the Vision Tower
To understand *why* these performance shifts occurred, I analyzed the internal attention of each model for specific failure and success cases. The methodology involved:
1. Identifying which frame received the most attention.
2. Analyzing the layer-wise attention over the frameq.
3. Extracting the spatial overlays on the actual frames in attended the most to visualize exactly *where* the model was looking.

<img width="1436" height="481" alt="image" src="https://github.com/user-attachments/assets/7cc4f075-36c7-4958-af3b-b517f0b1255d" />
*Example of success case for each models*

#### LLaVA-NeXT-Video (7B)
* **Attention Distribution:** Diffuse across layers and frames in the baseline. Selective attention successfully removes irrelevant tokens.
* **Architectural Details:** Features a spatiotemporal ViT-based architecture that enables attention within and across frames without explicit recurrent modules. Its CLIP/EVA ViT encoder has object-centric pretraining (100K VideoChatGPT-Instruct) for multi-step interactions, event-level dynamics, and object–action recognition, producing strong local embeddings.
* **Conclusion:** Selective attention leverages cross-frame integration and sequence-level priors. By removing irrelevant background noise, it forces focus onto key objects and interactions, ensuring intermediate actions are attended to. This directly improves temporal tracking and causal inference.

#### VideoLLaMA2 (7B)
* **Attention Distribution:** Naturally diffuse across frames and spatial regions, with a slight baseline bias toward salient objects. Selective attention slightly sharpens attention toward relevant objects, but the overall distribution remains diffuse.
* **Architectural Details:** Performs hierarchical spatial-temporal modeling, integrating information at both the frame and clip levels. It is trained on diverse video-text datasets (LLaVA-Instruct-150K, ShareGPT4V, VideoChat2-IT), enabling stable recognition of objects and actions across temporal spans.
* **Conclusion:** The hierarchical spatial-temporal design and robust local-global integration make the model largely insensitive to selective attention. It provides stable, but only modest gains.

#### mPLUG-Owl3 (7B)
* **Attention Distribution:** Globally distributed, often focusing on the agent. Selective attention reduces access to global anchors, producing patchy and unstable attention maps, which forces the model to hallucinate plausible but incorrect narratives.
* **Architectural Details:** Visual tokens are projected into a language-guided semantic space and fused with the LLM via hyper-attention blocks. It is optimized to create scene-level and sequence-level “visual anchors” for the LLM to reason about, rather than utilizing local object-centric processing.
* **Conclusion:** The model is designed for long-sequence summarization through anchor aggregation. Both selective attention (cropping) and additional frames reduce the fidelity of transient cues. This forces the model to compensate with linguistic priors, leading to a severe decrease in temporal and causal reasoning.


| Model | Architecture | Effect of Selective Attention | Hallucinations |
| :--- | :--- | :---: | :---: |
| **LLaVA-NeXT (7B)** | Object-Centric (ViT) | **Highest Benefit** ✅ | Decreased 📉 |
| **VideoLLaMA2 (7B)** | Hierarchical (S-T) | **Neutral Benefit** ➖ | Increased 📈 |
| **mPLUG-Owl3 (7B)** | Scene-Centric (Anchors) | **No Benefit** ❌ | Increased 📈 |



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
Each model (LLaVA, mPLUG, Video-LLaMA2) requires specific library versions to handle their respective attention mechanisms and vision towers.
Standard Base: Use the provided requirements.txt.
Model-Specific: The Section 1 of each Notebook contains the exact import needed, follow the instruction and download them to ensure all dependencies (Liger kernels, ModelScope, etc.) are correctly aligned with the architecture.
Run the specific Model_xxx.ipynb for the model you wish to test. Ensure you create different environments for each models, avoiding dependencies conflicts.

## Framework Explanation
### Phase 1: Biologically-Inspired Preprocessing (Manual.ipynb)
This notebook provides a custom interface to guide the segmentation process:
 - **Interactive Selection:** Add/remove points to define objects of interest.(points will create a bounding box to the object, that will be display right after the selection)
 - **Visualization:** Navigates across video frames or different clips.
 - **Automated Tracking:** Leverages SAM to track and segment causal agents across all frames.
 - **Output Generation:** Saves .npy masks, bounding boxes, and IDs for the next phase.
  
Every time a segmentation is performed the code saves:
- the frames (for visualization)
- the frames with the bouding boxes (for visualization)
- the frames with the masks (for visualization)
- the segmentation masks (.npy files used for Model inference process)
- the reconstructed video with the bounding boxes , IDs and masks
- the reconstructed video" with overlays on the objects only
- the tracking results for each objects accross frames
- the selections made (coordinate points to select an object for segmentation)

### Phase 2: Dual-Mode Inference (Model_xxx.ipynb)
The models are evaluated under two conditions:
1. **Original Mode (Baseline):** The VLM processes the video as-is.
2. **Crop-and-Mask:** Based on the generated masks, the video is cropped to isolate the objects of interest and zero-out the background, simulating foveal focus and removing environmental bias.

**Inference Options:**
- **Single Video Analysis:** Deep dive into a specific clip with real-time scoring.
- **Batch Analysis:** Automated run over the entire dataset, generating Description_table.csv and Summary_table.csv.

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
