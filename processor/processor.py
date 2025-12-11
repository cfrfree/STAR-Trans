import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss
from loss.cmt_loss import ModalityConsistencyLoss
from loss.topo_loss import TopologicalConsistencyLoss


def do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    # train pair
    if cfg.MODEL.PAIR:
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model.module.train_with_pair()
        else:
            model.train_with_pair()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            model.train()
            if cfg.MODEL.DIST_TRAIN and hasattr(train_loader_pair.sampler, "set_epoch"):
                train_loader_pair.sampler.set_epoch(epoch)
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_pair):
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                with amp.autocast(enabled=True):
                    logits_per_sar = model(img, target, cam_label=target_cam)
                    loss = clip_loss(logits_per_sar)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                        logger.info(
                            "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                                epoch,
                                (n_iter + 1),
                                len(train_loader_pair),
                                loss_meter.avg,
                                scheduler._get_lr(epoch)[0],
                            )
                        )

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch,
                        time_per_batch,
                        train_loader_pair.batch_size / time_per_batch,
                    )
                )

            if epoch % checkpoint_period == 0:
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
                    )


def do_train(
    cfg,
    model,
    center_criterion,
    train_loader,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    patch_size = 16

    h_num = cfg.INPUT.SIZE_TRAIN[0] // patch_size
    w_num = cfg.INPUT.SIZE_TRAIN[1] // patch_size

    cmt_loss_fn = ModalityConsistencyLoss()
    topo_loss_fn = TopologicalConsistencyLoss(h_num=h_num, w_num=w_num)

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_base_meter = AverageMeter()
    loss_cyc_meter = AverageMeter()
    loss_topo_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        loss_meter.reset()
        acc_meter.reset()
        loss_base_meter.reset()
        loss_cyc_meter.reset()
        loss_topo_meter.reset()

        evaluator.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, img_wh) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            img_wh = img_wh.to(device)

            with amp.autocast(enabled=True):
                outputs = model(img, target, cam_label=target_cam, img_wh=img_wh)

                loss_cyc = torch.tensor(0.0).to(device)
                loss_topo = torch.tensor(0.0).to(device)
                loss_base = torch.tensor(0.0).to(device)

                if isinstance(outputs, tuple) and len(outputs) == 5:
                    f_final, f_comp, cls_score, saliency, attn_map = outputs

                    f_final_list = [f_final[:, i, :] for i in range(f_final.size(1))]
                    loss_base = loss_fn(cls_score, f_final_list, target, target_cam)

                    if f_comp is not None:
                        loss_cyc = cmt_loss_fn(
                            f_final,
                            f_comp,
                            target,
                            target_cam,
                            saliency_scores=saliency,
                        )

                    if attn_map is not None:
                        loss_topo = topo_loss_fn(attn_map)

                    loss = loss_base + cfg.MODEL.CYC_LOSS_WEIGHT * loss_cyc + cfg.MODEL.TOPO_LOSS_WEIGHT * loss_topo

                else:
                    cls_score, feat = outputs
                    loss_base = loss_fn(cls_score, feat, target, target_cam)
                    loss = loss_base

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(cls_score, list):
                acc = (cls_score[0].max(1)[1] == target).float().mean()
            else:
                acc = (cls_score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            loss_base_meter.update(loss_base.item(), img.shape[0])
            loss_cyc_meter.update(loss_cyc.item(), img.shape[0])
            loss_topo_meter.update(loss_topo.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                            epoch,
                            (n_iter + 1),
                            len(train_loader),
                            loss_meter.avg,
                            acc_meter.avg,
                            scheduler._get_lr(epoch)[0],
                        )
                    )

        if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
            log_msg = f"Epoch {epoch} done. "

            log_msg += "\n   Stats: Acc: {:.2%}, Total Loss: {:.4f}, Base Loss: {:.4f}".format(acc_meter.avg, loss_meter.avg, loss_base_meter.avg)

            if cfg.MODEL.USE_CMT:
                log_msg += ", Cyc Loss: {:.4f}, Topo Loss: {:.4f}".format(loss_cyc_meter.avg, loss_topo_meter.avg)

            logger.info(log_msg)

        if epoch % checkpoint_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)),
                )

        if epoch % eval_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                model.eval()
                for n_iter, (
                    img,
                    vid,
                    camid,
                    camids,
                    target_view,
                    _,
                    img_wh,
                ) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
