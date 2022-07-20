# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import builtins
from collections import OrderedDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.engine import default_argument_parser, default_setup, launch
from fastreid.modeling import build_model
#import fastreid.modeling.moco as moco
from fastreid.modeling.moco import builder
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.evaluation import inference_on_dataset, print_csv_format, ReidEvaluator
from fastreid.utils.checkpoint import Checkpointer, PeriodicCheckpointer
from fastreid.utils import comm
from fastreid.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter
)

logger = logging.getLogger("fastreid")


def do_test(cfg, model):
    return NotImplementedError


def do_train(cfg, model, resume=False):
    data_loader = build_reid_train_loader(cfg)

    model.train()
    optimizer = build_optimizer(cfg, model)

    iters_per_epoch = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH
    scheduler = build_lr_scheduler(cfg, optimizer, iters_per_epoch)

    checkpointer = Checkpointer(
        model,
        cfg.OUTPUT_DIR,
        save_to_disk=comm.is_main_process(),
        optimizer=optimizer,
        **scheduler
    )

    start_epoch = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("epoch", -1) + 1
    )
    iteration = start_iter = start_epoch * iters_per_epoch

    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iter = max_epoch * iters_per_epoch
    warmup_iters = cfg.SOLVER.WARMUP_ITERS
    delay_epochs = cfg.SOLVER.DELAY_EPOCHS

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_epoch)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR)
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support some hooks, such as
    # accurate timing, FP16 training and precise BN here,
    # because they are not trivial to implement in a small training loop
    logger.info("Start training from epoch {}".format(start_epoch))
    with EventStorage(start_iter) as storage:
        for epoch in range(start_epoch, max_epoch):
            storage.epoch = epoch
            for data, _ in zip(data_loader, range(iters_per_epoch)):
                storage.iter = iteration

                loss_dict = model(data)
                print(loss_dict)
                aa
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

                if iteration - start_iter > 5 and (
                        (iteration + 1) % 200 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()

                iteration += 1

                if iteration <= warmup_iters:
                    scheduler["warmup_sched"].step()

            # Write metrics after each epoch
            for writer in writers:
                writer.write()

            if iteration > warmup_iters and (epoch + 1) >= delay_epochs:
                scheduler["lr_sched"].step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and epoch != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage

            periodic_checkpointer.step(epoch)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(gpu, args):
    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    # suppress printing if not master
    if gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    cfg = setup(args)
    arch_q = build_model(cfg)
    arch_k = build_model(cfg)
    model = builder.MoCo(arch_q, arch_k, 128, 65536, 0.999, 0.7, False)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model.cuda(gpu)
        model = DistributedDataParallel(model, device_ids=[gpu])

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    ), 
