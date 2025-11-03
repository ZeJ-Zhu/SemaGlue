import torch

from ..utils.tools import get_class
from .eval_pipeline import EvalPipeline


def get_benchmark(benchmark):#benchmarks:'hpatches'
    return get_class(f"{__name__}.{benchmark}", EvalPipeline)


@torch.no_grad()
#benchmark:要运行的基准测试的名称
def run_benchmark(benchmark, eval_conf, experiment_dir, model=None):
    """This overwrites existing benchmarks"""
    experiment_dir.mkdir(exist_ok=True, parents=True)
    #获取指定名称的基准测试对象
    bm = get_benchmark(benchmark)#<class 'gluefactory.eval.hpatches.HPatchesPipeline'>
    
    pipeline = bm(eval_conf)
    return pipeline.run(
        experiment_dir, model=model, overwrite=True, overwrite_eval=True
    )
