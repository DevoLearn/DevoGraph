## 1. Configure the environment   
conda create --name cell-tracking-challenge --file requirements-conda.txt   
conda activate cell-tracking-challenge     

## 2. Install other requirements      
pip install -r requirements.txt   

## 2. Download datasets   
http://celltrackingchallenge.net/2d-datasets/   

## 3. Put the download data to data directory look like:    

├── data                    <- Project data   
│   │   │   │   ├── 01                                    <- Sequence 01   
│   │   │   │   ├── 01_GT                                 <- Sequence 01 GT   
│   │   │   │   │   ├── TRA                                   <- Tracking GT   
│   │   │   │   ├── 02                                    <- Sequence 02   
│   │   │   │   ├── 02_GT                                 <- Sequence 02 GT   
│   │   │   │   ├── 01                                    <- Sequence 01   
│   │   │   │   ├── 01_GT                                 <- Sequence 01 GT   
│   │   │   │   │   ├── TRA                                   <- Tracking GT   
│   │   │   │   │   └── SEG                                   <- Tracking SEG (Not used)   
│   │   │   │   ├── 01_ST                                 <- Sequence 01 Silver GT   
│   │   │   │   └── └── SEG                                   <- Tracking SEG   
│   │   │   .   
│   │   │   .   
│   │   │   .   
│   │   ├── Test                             <- Graph Dataset implementation   
│   │   │   ├── Fluo-N2DH-SIM+                        <- Fluo-N2DH-SIM+ Dataset   
│   │   │   │   ├── 01                                    <- Sequence 01   
│   │   │   │   ├── 02                                    <- Sequence 02   
│   │   │   ├── PhC-C2DH-U373                             <- PhC-C2DH-U373 Dataset   
│   │   │   │   ├── 01                                    <- Sequence 01   
│   │   │   │   ├── 02                                    <- Sequence 02   
│   │   │   .   
│   │   │   .   
│   │   │   .   

## 4. Use the api to create csv   

import api_stage1
    #way1:use the default path (path message stored in stage1.py)   
    api_stage1.image2csv().create_csv_with_default()   
    #way2:pass params to function image2csv    
    api_stage1.image2csv(input_images, input_masks, input_seg,   
               input_model, output_csv, basic,   
               sequences, seg_dir).create_csv_with_default()   

## 5. The output csv stored in data/basic_features   
