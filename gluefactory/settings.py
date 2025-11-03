from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
#DATA_PATH = root / "data/"  # datasets and pretrained weights
DATA_PATH = "/media/suhzhang/dataset/MatcherNet/glue_data/"# datasets and pretrained weights
DATA_DINO_PATH = "/data1/zzj/exports/"
SEGMENT_PATH = "/data/suhzhang/dataset/zzj/" #segment 特征图
SEGMENT_F2_PATH = "/data1/zzj/" #不固定的
SEGMENT_F2_PATH_C = "/media/suhzhang/dataset/MatcherNet/glue_data/"
SEGMENT_F2_MEGA_PATH_TRAIN ="/media/suhzhang/dataset/MatcherNet/glue_data/"
SEGMENT_F2_MEGA_PATH_VAL = "/media/suhzhang/dataset/MatcherNet/glue_data/"
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
TRAIN_VAL_DATA_PATH="/media/suhzhang/dataset/MatcherNet/glue_data/"#revisitop1m数据集