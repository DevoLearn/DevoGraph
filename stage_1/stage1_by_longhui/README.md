#1.configure the environment
conda create --name cell-tracking-challenge --file requirements-conda.txt
conda activate cell-tracking-challenge
# install other requirements
pip install -r requirements.txt

#2.download datasets
http://celltrackingchallenge.net/2d-datasets/

#3.put the download data to data directory look like:
├── data                    <- Project data
│   │   │   │   ├── 01                                    <- Seuqence 01
│   │   │   │   ├── 01_GT                                 <- Seuqence 01 GT
│   │   │   │   │   ├── TRA                                   <- Tracking GT
│   │   │   │   ├── 02                                    <- Seuqence 02
│   │   │   │   ├── 02_GT                                 <- Seuqence 02 GT
│   │   │   │   ├── 01                                    <- Seuqence 01
│   │   │   │   ├── 01_GT                                 <- Seuqence 01 GT
│   │   │   │   │   ├── TRA                                   <- Tracking GT
│   │   │   │   │   └── SEG                                   <- Tracking SEG (Not used)
│   │   │   │   ├── 01_ST                                 <- Seuqence 01 Silver GT
│   │   │   │   └── └── SEG                                   <- Tracking SEG
│   │   │   .
│   │   │   .
│   │   │   .
│   │   ├── Test                             <- Graph Dataset implementation
│   │   │   ├── Fluo-N2DH-SIM+                        <- Fluo-N2DH-SIM+ Dataset
│   │   │   │   ├── 01                                    <- Seuqence 01
│   │   │   │   ├── 02                                    <- Seuqence 02
│   │   │   ├── PhC-C2DH-U373                             <- PhC-C2DH-U373 Dataset
│   │   │   │   ├── 01                                    <- Seuqence 01
│   │   │   │   ├── 02                                    <- Seuqence 02
│   │   │   .
│   │   │   .
│   │   │   .

#3.
#4.use the api to create csv

import api_stage1
    #way1:use the default path (path message stored in stage1.py)
    api_stage1.image2csv().create_csv_with_default()
    #way2:pass params to function image2csv 
    api_stage1.image2csv(input_images, input_masks, input_seg,
               input_model, output_csv, basic,
               sequences, seg_dir).create_csv_with_default()

#5.the output csv stored in data/basic_features
