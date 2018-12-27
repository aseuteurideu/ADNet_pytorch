# matlab code:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_train_videos.m

from options.general import opts
from utils.get_benchmark_path import get_benchmark_path
from utils.get_benchmark_info import get_benchmark_info
import numpy as np
import numpy.matlib

def get_train_videos(opts):
    train_db_names = opts['train_dbs']
    test_db_names = opts['test_db']

    video_names = []
    video_paths = []
    bench_names = []

    for dbidx in range(len(train_db_names)):
        bench_name = train_db_names[dbidx]
        path_ = get_benchmark_path(bench_name)
        video_names_ = get_benchmark_info(train_db_names[dbidx] + '-' + test_db_names)
        video_paths_ = np.matlib.repmat(path_, 1, len(video_names_))
        video_names.extend(video_names_)
        video_paths.extend(list(video_paths_[0]))
        bench_names.extend(list(np.matlib.repmat(bench_name, 1, len(video_names_))[0]))

    train_db = {
        'video_names' : video_names,
        'video_paths' : video_paths,
        'bench_names' : bench_names
    }
    return train_db

# test the module
#get_train_videos(opts)