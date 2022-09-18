# DevoGraph
## Introduction
* DevoGraph is a [GSoC 2022 project](https://neurostars.org/t/gsoc-2022-project-idea-gnns-as-developmental-networks/21368) under the administration of [INCF](https://www.incf.org/) and [DevoWorm](https://devoworm.weebly.com/). Our main goal is to provide examples and components that utlize (Temporal/Directed/...) Graph Neural Networks to model the developmental process of *[C. elegans](https://en.wikipedia.org/wiki/Caenorhabditis_elegans)*. 

## Developers
* GSoC 2022 participants: [Jiahang Li](https://github.com/LspongebobJH), [Wataru Kawakami](https://github.com/watarungurunnn)
* Mentors: [Bradly Alicea](https://bradly-alicea.weebly.com/), [Jesse Parent](https://jesparent.github.io/)
* External contributors: [Longhui Jiang](https://github.com/jianglonghui)

## Contributions
### Jiahang Li
* Design a KNN-based method constructing ****temporal** graphs**. The method is implemented in `./devograph/datasets/datasets.py`. These temporal graphs are based on 3d positions of cell centroids and mimic cell developmental process of *C. elegans*. Each node represents a cell at a certain frame, and edges at the same frame connect neighbors according to KNN while edges across different frames connect mother and daughter cells. Please refer to `./stage_2/stage_2.ipynb` to check more details. 
* Refactor codes of constructing ****directed** graphs** initially implemented by [cell-track-gnn](https://github.com/talbenha/cell-tracker-gnn). The re-implementation is in `./devograph/datasets/datasets1.py`. This method gives each edge an direction implying the relationship between mother and daughter cells.
* Refactor codes of a **directed GNN** initially implemented by [cell-track-gnn](https://github.com/talbenha/cell-tracker-gnn). The re-implementation is in `./devograph/models/ct.py`. The GNN is based on directed graphs and incorporates information of nodes and edges to aggregate messages.
* Both of re-implementations above abstract the core logic, remove redundant and unrelated codes and unnecessary third-party frameworks, and finally provide easy-to-use APIs.
* Design the whole pipeline of DevoGraph presented in `./miscellaneous/GSoC 2022 22.1.pdf`.
* Assign tasks to other participants.

### Wataru Kawakami
* Please refer to [Wataru Kawakami](https://github.com/watarungurunnn/GSoC2022_submission) to check his contributions.

### Longhui Jiang
* Refactor codes of pre-processing 2-D images(frames of videos) and converting them into location information of cells stored in csv files. The re-implementation is based on [cell-track-gnn](https://github.com/talbenha/cell-tracker-gnn). Please refer to `./stage_1/` and `./stage_1/stage1_by_longhui` for more details.
* Note that in `./stage_1/`, `./stage_1/stage1_by_longhui` is implemented by Longhui and other files are implemented by Wataru.
