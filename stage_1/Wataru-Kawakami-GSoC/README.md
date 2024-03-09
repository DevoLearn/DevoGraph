# GSoC2022_submission

## Our (future) goal
* Automated high throughput cell tracking by GNN

## Motivations
* Cell tracking of organisms will help us understand the mechanisms under development and cell-cell interactions.
* Cells exist in relationships with each other, so the cell location are affected by topological factors, which might be expressed by graph and GNN.

## What was done
1. Searched for previous researches and discussed with the mentors and contributors which graph expression and GNN structure are effective for cell tracking.
2. Refactored inference part of the source codes of: Ben-Haim, T. & Riklin-Raviv, T. Graph Neural Network for Cell Tracking in Microscopy Videos. Preprint at http://arxiv.org/abs/2202.04731 (2022).
4. Implemented preprocessing 3D microscopy videos of C.elegans into centroids graph by devolearn.
5. Fixed bugs of external modules.

## Details of what was done
### 1. Searched for previous researches and discussed with the mentors and contributors which graph expression and GNN structure are effective for cell tracking.
Our project has started as a research project and we spent a lot of time searching for previous researches and discussing which features of the target cells and which type of GNN structure are useful for effectively capture the development of organisms.
We had many difficulties in the preprocessing part, so ran out of time to implement and test GNNs.
Here I summarize current idea about the characteristics of the cells, difficulties, and realistic approach.

Characteristics:
* Each cell occupies some spaces and the locations of the centroids are decided by interactions between nearby cells.
* These interactions are assumed to be expressed by edges of the graphs, while the centroids of the cells are expressed by nodes.

Difficulties:
* The mitosis, overlaps and occlusions, visual similarity, change in appearance make it difficult to accurately identify the cells.
* Cell dynamics appear completely random, while pedestrians and cars have somehow fixed directions.
* The features of the microscopy videos vary so much that complex preprocessing is necessary for each of the datasets.
* Manual annotations are required to obtain training datasets, so few datasets are available.

Future plan:
* Apply to cell tracking Spatio-Temporal Graph Transformer used for tasks such as pedestrian tracking.
  * Pedestrian keeps some distance from each other, which could be an analogy of the fact that cell nucleus keep distance from each other, separated by membrane.
  * As mentioned above, The mitosis and overlaps make it more difficult than other tracking tasks.
  * Reference: Yu, C., Ma, X., Ren, J., Zhao, H. & Yi, S. Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction. Preprint at https://doi.org/10.48550/arXiv.2005.08514 (2020).


### 2. Refactored inference part of the source codes of: Ben-Haim, T. & Riklin-Raviv, T. Graph Neural Network for Cell Tracking in Microscopy Videos. Preprint at http://arxiv.org/abs/2202.04731 (2022).
- Added [codes for segmentation](https://github.com/watarungurunnn/cell-tracker-gnn/tree/main/src/inference/segmentation) to enable users to make graph from the source video datasets. [PR under review](https://github.com/jianglonghui/cell-tracker-gnn/pull/1) (-> [screenshot](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2020.00.27%20PM.png))
  - The cell-tracker-gnn lacked examples of segmentation process so it was difficult for end-users to try inference from videos.
- Refactored [inference part](https://github.com/watarungurunnn/cell-tracker-gnn/tree/main/src/inference) of the source code to make it easy for end-users to use the trained model to make graph from microscopy videos. [PR under review](https://github.com/jianglonghui/cell-tracker-gnn/pull/1) (-> [screenshot](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2020.00.27%20PM.png)
  - Defined config file for inference part to allow users to easily input minimum amount of parameters required for inference.
  - Refactored based on object oriented programming to make future edits easy.


### 3. Implemented preprocessing 3D microscopy videos of C.elegans into centroids graph by devolearn.
- [Reported](https://github.com/LspongebobJH/DevoGraph/blob/wataru/stage_1/stage_1/stage_1.ipynb) ([commit](https://github.com/LspongebobJH/DevoGraph/commit/4b88c23f2fb9c4da7f633de5a94f946d4176d46e)) that devolearn does not generalize to other datasets. -> future work
  - [PR merged](https://github.com/LspongebobJH/DevoGraph/pull/1), [PR under review](https://github.com/LspongebobJH/DevoGraph/pull/5) (-> [screenshot1](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2016.38.59%20PM.png))     (-> [screenshot2](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2019.55.45%20PM.png))
- Added devolearn nucleus segmentor for video. [here](https://github.com/DevoLearn/devolearn/pull/74).   (-> [screenshot](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2011.45.27.png))


### 4. Fixed bugs of external modules.
- devolearn - Installed package from the pip could not used as written in README.md, so added instruction to install from the source. [PR under review](https://github.com/DevoLearn/devolearn/pull/73) (-> [screenshot](https://github.com/watarungurunnn/GSoC2022_submission/blob/main/Screen%20Shot%202022-09-12%20at%2019.48.13%20PM.png)
- devolearn - Fixed wrong URL of the provided trained model. [PR merged](https://github.com/DevoLearn/devolearn/pull/67/commits/55356cdbdcd0d89e16631f883b40b0cc35f1ca13)
- scikit-image - Fixed a function of "regionprops" of scikit-image, which requires non-negative input, to consider negative value input very close to 0 to be 0. [Folk](https://github.com/scikit-image/scikit-image/commit/dede59c19817bceccf80de8eb59eb29db746e1c5)
  - The negative values appeared to be computational error.
  - This change was needed to run the cell-tracking-gnn on 3D datasets.

### What has not been done
- Merge whole refactored codes of Ben-Haim, T. & Riklin-Raviv, T. Graph Neural Network for Cell Tracking in Microscopy Videos. Preprint at http://arxiv.org/abs/2202.04731 (2022). to provide complete API.
- Create effective GNN for cell tracking.
- Apply cell tracking result to other researches such as simulations.
