# scREC for matrix final project

## hyper-parameters

+ gamma
+ lamb
+ delta

## quick start
pip install -r requirements.txt

python run_reduction.py --task_name TASK_NAME --method METHOD --K 20

### TASK_NAME
+ tissue
+ forebrain
+ gmvshek
+ gmvshl
+ insilico

### METHOD
+ **sc-only**: use single cell data only
+ **bulk-only**: set sc_W same as bulk_W; update sc_H only
+ **bulk-sc**: use bulk data train result as references
+ **bulk-sc-aug**: data augmentation
