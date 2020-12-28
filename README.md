# matrix-project

## hyper-parameters

+ gamma
+ lamb
+ delta

## quick start
python run_reduction.py --task_name TASK_NAME --method METHOD

### TASK_NAME
+ tissue
+ forebrain
+ gmvshek
+ gmvshl
+ insilico

### METHOD
+ sc-only: use single cell data only
+ bulk-only: set sc_W same as bulk_W; update sc_H only
+ bulk-sc: use bulk data train result as references
+ bulk-sc-aug: data augmentation
