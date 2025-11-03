# import debugpy
# debugpy.listen(('127.0.13.25', 8002))
# debugpy.wait_for_client()

"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import argparse
import copy
import re#提供正则化表达式操作
import shutil#提供文件操作的高级工具，用于复制，移动和删除
import signal#提供与信号处理相关的操作
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
import torch
#print(torch.__file__)
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm#在命令行中显示进度条

from gluefactory import __module_name__, logger
from .datasets import get_dataset
from .eval import run_benchmark
from .models import get_model
from .settings import EVAL_PATH, TRAINING_PATH
from .scripts.export_segment import run_export_seg_homography
from .utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from .utils.stdout_capturing import capture_outputs
from .utils.tensor import batch_to_device
from .utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    set_seed,
)

# @TODO: Fix pbar pollution in logs
# @TODO: add plotting during evaluation

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
}
default_train_conf = OmegaConf.create(default_train_conf)


@torch.no_grad()
def do_evaluation(model, loader, device, loss_fn, conf, pbar=True):
    #将 PyTorch 模型设置为评估模式，这会关闭模型中的一些具有随机性质的层，例如 Dropout 层，以便在评估中保持一致性
    model.eval()
    #保存评估结果
    results = {}
    #保存precision-Recall曲线的度量
    pr_metrics = defaultdict(PRMetric)
    figures = []
    #在每个迭代中从数据加载器中随机选择一些样本进行可视化。可视化的数量由 n 决定，plot_fn 是用于生成可视化的函数
    if conf.plot is not None:
        n, plot_fn = conf.plot
        plot_ids = np.random.choice(len(loader), min(len(loader), n), replace=False)
    for i, data in enumerate(
        tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)
    ):
        data = batch_to_device(data, device, non_blocking=True)
        #在评估阶段，不进行梯度计算，以节省内存和计算资源
        with torch.no_grad():
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            #将可视化图表添加到 figures 中
            if conf.plot is not None and i in plot_ids:
                figures.append(locate(plot_fn)(pred, data))
            # add PR curves
            #针对 Precision-Recall 曲线的配置，更新 pr_metrics
            for k, v in conf.pr_curves.items():
                pr_metrics[k].update(
                    pred[v["labels"]],
                    pred[v["predictions"]],
                    mask=pred[v["mask"]] if "mask" in v.keys() else None,
                )
            del pred, data
        #统计并记录每个评估指标的平均值、中位数（如果在配置中指定了）和召回率（如果在配置中指定了）
        numbers = {**metrics, **{"loss/" + k: v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[k + "_median"] = MedianMetric()
                if k in conf.recall_metrics.keys():
                    q = conf.recall_metrics[k]
                    results[k + f"_recall{int(q)}"] = RecallMetric(q)
            results[k].update(v)
            if k in conf.median_metrics:
                results[k + "_median"].update(v)
            if k in conf.recall_metrics.keys():
                q = conf.recall_metrics[k]
                results[k + f"_recall{int(q)}"].update(v)
        del numbers
    results = {k: results[k].compute() for k in results}
    #返回计算后的结果，其中包括评估指标的平均值、Precision-Recall 曲线的度量以及可视化图表
    return results, {k: v.compute() for k, v in pr_metrics.items()}, figures


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf.train.lr_schedule."""
    if conf.type not in ["factor", "exp", None]:
        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            #使用 torch.optim.lr_scheduler.StepLR，其中学习率在每个步骤（epoch）之后按照 conf.factor 的倍数衰减
            # assert conf.light < conf.start
            # if it < conf.light:
            # elif it < conf.start:
            #     return conf.light_factor
            # else:
            #     return conf.factor
            # if it < conf.start:
            #     return 1.0
            # elif it < conf.end:
            #     return conf.factor
            # else: return 1.0
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            # 假设 conf.decay_rate 是一个正数，表示衰减速率
            # decay_rate = conf.decay_rate
            # # 计算当前迭代次数的学习率衰减系数
            # gam = 1 / (1 + decay_rate * (it-conf.start))#decay_rate= 4.3
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        # TODO: use proper regexp rather than just this inclusion check
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    lr_params = [
        {"lr": scale * base_lr, "params": [p for _, p in ps]}
        for scale, ps in scale2params.items()
    ]
    return lr_params


def training(rank, conf, output_dir, args):
    if args.restore:#判断是否从先前的训练中恢复，不恢复
        logger.info(f"Restoring from previous training of {args.experiment}")
        try:
            init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        except AssertionError:
            init_cp = get_best_checkpoint(args.experiment)
        # init_cp = "/data/zzj/glue-factory-main1/glue-factory-main/outputs/training/mega_dino_divide_21_ffn_nfnorm_NewSEncoder_loadfalse_lr20/checkpoint_3_22942.tar"
        # init_cp = "/data/zzj/glue-factory-main1/glue-factory-main/outputs/training/mega_dino_21_ffn_nfnorm_loadfalse_lr20/checkpoint_3_20000.tar"
        logger.info(f"Restoring from checkpoint {init_cp.name}")
        init_cp = torch.load(str(init_cp), map_location="cpu")
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = init_cp["epoch"] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location="cpu")
        best_eval = best_cp["eval"][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0#当前训练的轮次
        best_eval = float("inf")#最佳评估指标
        if conf.train.load_experiment:#为none
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            # the user has to make sure that the weights are compatible
            try:
                init_cp = get_last_checkpoint(conf.train.load_experiment)
            except AssertionError:
                init_cp = get_best_checkpoint(conf.train.load_experiment)
            # init_cp = "/data/zzj/glue-factory-main1/glue-factory-main/outputs/training/mega_aliked_divide_21_ffn_nfnorm_c_NewEncoder_loadfalse_lr20/checkpoint_best.tar"
            init_cp = torch.load(str(init_cp), map_location="cpu")
            # load the model config of the old setup, and overwrite with current config
            conf.model = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).model, conf.model
            )
            print(conf.model)
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:#进程编号rank
        writer = SummaryWriter(log_dir=str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f"Training in distributed mode with {args.n_gpus} GPUs")
        assert torch.cuda.is_available()
        device = rank
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.n_gpus,
            rank=device,
            init_method="file://" + str(args.lock_file),
        )
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if "batch_size" in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if "train_batch_size" in data_conf:
            data_conf.train_batch_size = int(data_conf.train_batch_size / args.n_gpus)
        if "num_workers" in data_conf:
            data_conf.num_workers = int(
                (data_conf.num_workers + args.n_gpus - 1) / args.n_gpus
            )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}")

    dataset = get_dataset(data_conf.name)(data_conf)

    # Optionally load a different validation dataset than the training one泛化
    #如果指定了验证集就用，没指定就用测试集作验证集，指定了
    val_data_conf = conf.get("data_val", None)                   
    if val_data_conf is None:
        val_dataset = dataset
    else:
        val_dataset = get_dataset(val_data_conf.name)(val_data_conf)

    # @TODO: add test data loader
    #根据不同的模式（过拟合模式或正常模式）创建相应的数据加载器，并提供一些有关数据加载器的信息，正常模式
    if args.overfit:#走正常模式
        # we train and eval with the same single training batch
        logger.info("Data in overfitting mode")
        assert not args.distributed
        train_loader = dataset.get_overfit_loader("train")
        val_loader = val_dataset.get_overfit_loader("val")
    else:
        train_loader = dataset.get_data_loader("train", distributed=args.distributed)
        val_loader = val_dataset.get_data_loader("val")
    if rank == 0:
        logger.info(f"Training loader has {len(train_loader)} batches")
        logger.info(f"Validation loader has {len(val_loader)} batches")

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):#捕获键盘中断信号，根据标志stop停止训练
        logger.info("Caught keyboard interrupt signal, will terminate")
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True

    stop = False
    signal.signal(signal.SIGINT, sigint_handler)
    model = get_model(conf.model.name)(conf.model).to(device)#创建模型实例,
    if args.compile:
        model = torch.compile(model, mode=args.compile)
    loss_fn = model.loss
    if init_cp is not None:
        model.load_state_dict(init_cp["model"], strict=False)
        # 过滤掉不匹配的参数
        # state_dict = init_cp["model"]
        # model_dict = model.state_dict()
        
        # pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        # model_dict.update(pretrained_dict)
        
        # model.load_state_dict(model_dict, strict=False)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    if rank == 0 and args.print_arch:#应该没执行
        logger.info(f"Model: \n{model}")

    torch.backends.cudnn.benchmark = True#启用了cuda计算优化选项
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {#优化器和学习率调度器设置
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    
    if init_cp is not None:
        par_names = [name for name, _ in init_cp["model"].items()]
        params_names = []
        for i in range(len(params)):
            params_names.append(params[i][0])
        par_names_unique = set(par_names) - set(params_names)

    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]

    lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(
        lr_params, lr=conf.train.lr, **conf.train.optimizer_options
    )
    scaler = GradScaler(enabled=args.mixed_precision is not None)#设置混合精度训练的数据类型
    logger.info(f"Training with mixed_precision={args.mixed_precision}")#Training with mixed_precision=None

    mp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: torch.float32,  # we disable it anyway
    }[args.mixed_precision]

    results = None  # fix bug with it saving

    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)
    if args.restore:
        # optimizer.load_state_dict(init_cp["optimizer"])
        # if "lr_scheduler" in init_cp:
        #     lr_scheduler.load_state_dict(init_cp["lr_scheduler"])

        try:
            blank_optim_state_dict = optimizer.state_dict()
            # 获取加载
            opt_state_dict = init_cp["optimizer"]
            for k, v in opt_state_dict["param_groups"][0].items():
                if k != "params":
                    # 将加载数据转移到模板中
                    blank_optim_state_dict["param_groups"][0][k] = v
            for k, state in opt_state_dict["state"].items():
                blank_optim_state_dict["state"][k] = state
            optimizer.load_state_dict(blank_optim_state_dict)

            # 加载学习率调度器状态
            if "lr_scheduler" in init_cp:
                lr_scheduler.load_state_dict(init_cp["lr_scheduler"])
        except ValueError as e:
            print(f"Error loading optimizer state: {e}")
            # 重新初始化优化器
            optimizer = optimizer_fn(model.parameters(), lr=conf.train.lr)
        
    if rank == 0:#在编号为0的进程上输出配置信息
        logger.info(
            "Starting training with configuration:\n%s", OmegaConf.to_yaml(conf)
        )
    losses_ = None

    def trace_handler(p):
        # torch.profiler.tensorboard_trace_handler(str(output_dir))
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
        p.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
    while epoch < conf.train.epochs and not stop:
        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        # we first run the eval
        #当epoch能够整除conf.train.test_every_epoch且args.run_benchmarks为真时，执行评估
        if (
            rank == 0
            and epoch % conf.train.test_every_epoch == 0
            and args.run_benchmarks
        ):
            for bname, eval_conf in conf.get("benchmarks", {}).items():
                logger.info(f"Running eval on {bname}")
                #run_benchmark函数运行具体的评估，并得到评估结果
                s, f, r = run_benchmark(
                    bname,
                    eval_conf,
                    EVAL_PATH / bname / args.experiment / str(epoch),#根据当前epoch和评估配置，设置评估结果的保存路径
                    model.eval(), # 将模型设置为评估模式
                )
                logger.info(str(s))
                for metric_name, value in s.items():
                    writer.add_scalar(f"test/{bname}/{metric_name}", value, epoch)
                # for fig_name, fig in f.items():
                #     writer.add_figure(f"figures/{bname}/{fig_name}", fig, epoch)
                del s, f

        # set the seed
        set_seed(conf.train.seed + epoch)

        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step()
            logger.info(
                f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}'
            )
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn and not args.overfit:
            loaders = [train_loader]
            if conf.train.dataset_callback_on_val:
                loaders += [val_loader]
            for loader in loaders:
                if isinstance(loader.dataset, torch.utils.data.Subset):
                    getattr(loader.dataset.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
                else:
                    getattr(loader.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
        
        for it, data in enumerate(train_loader):#data表示当前批次的训练数据  调用了_get_item_
            #用当前迭代索引it和epoch计算总迭代次数tot_it，并计算tot_n_samples
            tot_it = (len(train_loader) * epoch + it) * (
                args.n_gpus if args.distributed else 1
            )
            tot_n_samples = tot_it
            if not args.log_it:
                # We normalize the x-axis of tensorflow to num samples!
                tot_n_samples *= train_loader.batch_size

            model.train()#将模型设置为训练模式
            optimizer.zero_grad()#清零模型参数

            #run_export_seg_homography(val_loader)
            with autocast(enabled=args.mixed_precision is not None, dtype=mp_dtype):#以较低精度的数据类型执行操作
                data = batch_to_device(data, device, non_blocking=True)
                #run_export_seg_homography(train_loader)
                pred = model(data)
                losses, _ = loss_fn(pred, data)
                loss = torch.mean(losses["total"])
            # for name, param in model.matcher.named_parameters():
            #     if param.grad is None:
            #         print(name)
            if torch.isnan(loss).any():
                print(f"Detected NAN, skipping iteration {it}")
                del pred, data, loss, losses
                continue
            #do_backward根据损失是否需要梯度计算得到
            do_backward = loss.requires_grad
            #将do_backward转换为张量并通过全局归约操作将其设置为True或False
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(
                    do_backward, torch.distributed.ReduceOp.PRODUCT
                )
                do_backward = do_backward > 0
            if do_backward:#反向传递和梯度缩放
                scaler.scale(loss).backward()
                #如果检测到异常梯度
                if args.detect_anomaly:
                    # Check for params without any gradient which causes
                    # problems in distributed training with checkpointing
                    detected_anomaly = False
                    for name, param in model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            print(f"param {name} has no gradient.")
                            detected_anomaly = True
                    if detected_anomaly:
                        raise RuntimeError("Detected anomaly in training.")
                #如果配置了梯度裁剪
                if conf.train.get("clip_grad", None):
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            all_params,
                            max_norm=conf.train.clip_grad,
                            error_if_nonfinite=True,
                        )
                        scaler.step(optimizer)
                    except RuntimeError:
                        logger.warning("NaN detected in gradients. Skipping iteration.")
                    scaler.update()
                else:
                    scaler.step(optimizer)#执行优化器
                    scaler.update()#执行梯度更新
                #如果学习率调度不是基于epoch的
                if not conf.train.lr_schedule.on_epoch:
                    lr_scheduler.step()
            #如果do_backward为False，记录一个警告信息，表示由于detach操作，跳过反向传播
            else:
                #如果不需要进行反向传播，记录警告信息
                if rank == 0:
                    logger.warning(f"Skip iteration {it} due to detach.")

            #启用性能分析
            if args.profile:
                prof.step()

            if it % conf.train.log_every_iter == 0:
                for k in sorted(losses.keys()):
                    if args.distributed:
                        losses[k] = losses[k].sum(-1)
                        torch.distributed.reduce(losses[k], dst=0)
                        losses[k] /= train_loader.batch_size * args.n_gpus
                    losses[k] = torch.mean(losses[k], -1)
                    losses[k] = losses[k].item()
                if rank == 0:#在 rank 为 0 的进程上记录损失信息，包括输出到日志中和使用 TensorBoard 进行可视化
                    str_losses = [f"{k} {v:.3E}" for k, v in losses.items()]
                    logger.info(
                        "[E {} | it {}] loss {{{}}}".format(
                            epoch, it, ", ".join(str_losses)
                        )
                    )
                    for k, v in losses.items():
                        writer.add_scalar("training/" + k, v, tot_n_samples)
                    writer.add_scalar(
                        "training/lr", optimizer.param_groups[0]["lr"], tot_n_samples
                    )
                    writer.add_scalar("training/epoch", epoch, tot_n_samples)

            #是否记录梯度信息
            if conf.train.log_grad_every_iter is not None:#none
                if it % conf.train.log_grad_every_iter == 0:
                    grad_txt = ""
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            if name.endswith("bias"):
                                continue
                            writer.add_histogram(
                                f"grad/{name}", param.grad.detach(), tot_n_samples
                            )
                            norm = torch.norm(param.grad.detach(), 2)
                            grad_txt += f"{name} {norm.item():.3f}  \n"
                    writer.add_text("grad/summary", grad_txt, tot_n_samples)
            del pred, data, loss, losses#释放资源

            # Run validation
            if (
                (
                    it % conf.train.eval_every_iter == 0
                    and (it > 0 or epoch == -int(args.no_eval_0))
                )
                or stop
                or it == (len(train_loader) - 1)
            ):
                # 执行验证的条件：
                # 1. 每隔一定的迭代次数执行一次验证（eval_every_iter）
                # 2. 在第一次迭代之后，或者在 epoch 为指定值时执行验证
                # 3. 或者在停止标志（stop）为真时执行验证
                # 4. 或者在遍历完训练数据集的最后一批时执行验证
                # 在执行验证之前，通过 fork_rng 设置验证时的随机数种子
                with fork_rng(seed=conf.train.seed):
                    results, pr_metrics, figures = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        pbar=(rank == -1),
                    )

                #只在主进程上记录验证结果
                if rank == 0:
                    #将验证结果以及评估指标记录
                    str_results = [
                        f"{k} {v:.3E}"
                        for k, v in results.items()
                        if isinstance(v, float)
                    ]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')
                    #将验证结果记录到tensorboard
                    for k, v in results.items():
                        if isinstance(v, dict):
                            writer.add_scalars(f"figure/val/{k}", v, tot_n_samples)
                        else:
                            writer.add_scalar("val/" + k, v, tot_n_samples)
                    #将Precision-Recall曲线记录到TensorBoard
                    for k, v in pr_metrics.items():
                        writer.add_pr_curve("val/" + k, *v, tot_n_samples)
                    # @TODO: optional always save checkpoint
                    #如果验证结果是最好的，保存模型检查点
                    if results[conf.train.best_key] < best_eval:
                        best_eval = results[conf.train.best_key]
                        save_experiment(
                            model,
                            optimizer,
                            lr_scheduler,
                            conf,
                            losses_,
                            results,
                            best_eval,
                            epoch,
                            tot_it,
                            output_dir,
                            stop,
                            args.distributed,
                            cp_name="checkpoint_best.tar",
                        )
                        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
                    # if len(figures) > 0:
                    #     for i, figs in enumerate(figures):
                    #         for name, fig in figs.items():
                    #             writer.add_figure(
                    #                 f"figures/{i}_{name}", fig, tot_n_samples
                    #             )
                torch.cuda.empty_cache()  # should be cleared at the first iter
                del pr_metrics, figures

            if (tot_it % conf.train.save_every_iter == 0 and tot_it > 0) and rank == 0:
                if results is None:
                    results, _, _ = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        pbar=(rank == -1),
                    )
                    best_eval = results[conf.train.best_key]
                best_eval = save_experiment(
                    model,
                    optimizer,
                    lr_scheduler,
                    conf,
                    losses_,
                    results,
                    best_eval,
                    epoch,
                    tot_it,
                    output_dir,
                    stop,
                    args.distributed,
                )

            if stop:
                break

        if rank == 0:
            best_eval = save_experiment(
                model,
                optimizer,
                lr_scheduler,
                conf,
                losses_,
                results,
                best_eval,
                epoch,
                tot_it,
                output_dir=output_dir,
                stop=stop,
                distributed=args.distributed,
            )

        epoch += 1

    logger.info(f"Finished training on process {rank}.")
    if rank == 0:
        writer.close()


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(output_dir / "log.txt"):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment {args.experiment}")
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    elif args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))

    # copy gluefactory and submodule into output dir
    for module in conf.train.get("submodules", []) + [__module_name__]:
        mod_dir = Path(__import__(str(module)).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(conf, output_dir, args)
        )
    else:
        main_worker(0, conf, output_dir, args)