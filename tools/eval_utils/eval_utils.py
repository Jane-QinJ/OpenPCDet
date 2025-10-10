import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()   # model forward (+postproc in model)
        load_time_meter  = common_utils.AverageMeter()   # host->GPU copy
        pp_time_meter    = common_utils.AverageMeter()   # generate_prediction_dicts
        io_time_meter    = common_utils.AverageMeter()   # load + pp
        first_batch_size = None

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        # ----------------- measure load (H2D) -----------------
        if getattr(args, 'infer_time', False):
            load_start_time = time.time()
        load_data_to_gpu(batch_dict)
        if getattr(args, 'infer_time', False):
            if first_batch_size is None:
                first_batch_size = int(batch_dict.get('batch_size', 1))
            torch.cuda.synchronize()
            load_ms = (time.time() - load_start_time) * 1000.0
            load_time_meter.update(load_ms)

        # ----------------- model forward -----------------
        if getattr(args, 'infer_time', False):
            torch.cuda.synchronize()
            inference_start_time = time.time()
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        if getattr(args, 'infer_time', False):
            torch.cuda.synchronize()
            infer_ms = (time.time() - inference_start_time) * 1000.0
            infer_time_meter.update(infer_ms)

        disp_dict = {}

        # ----------------- post-process dicts -----------------
        if getattr(args, 'infer_time', False):
            pp_start_time = time.time()
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        if getattr(args, 'infer_time', False):
            # 有些 NMS/后处理在 GPU 上，确保同步
            torch.cuda.synchronize()
            pp_ms = (time.time() - pp_start_time) * 1000.0
            pp_time_meter.update(pp_ms)
            io_time_meter.update(load_ms + pp_ms)
            # 进度条同时显示：纯模型时间 和 数据搬运+前后处理时间
            disp_dict['infer_ms'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
            disp_dict['io_ms']    = f'{(load_ms+pp_ms):.2f}({io_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    if getattr(args, 'infer_time', False) and first_batch_size:
        avg_infer_ms = infer_time_meter.avg
        avg_load_ms  = load_time_meter.avg
        avg_pp_ms    = pp_time_meter.avg
        avg_io_ms    = io_time_meter.avg
        fps_model = first_batch_size / (avg_infer_ms / 1000.0)
        fps_e2e_model_plus_io = first_batch_size / ((avg_infer_ms + avg_io_ms) / 1000.0)
        logger.info(f'Model latency (avg): {avg_infer_ms:.2f} ms per batch of {first_batch_size}, FPS(model): {fps_model:.2f}')
        logger.info(f'Data+Pre/Post latency (avg): load {avg_load_ms:.2f} ms + pp {avg_pp_ms:.2f} ms = {avg_io_ms:.2f} ms per batch')
        logger.info(f'Combined latency (model + data/pre/post): {(avg_infer_ms + avg_io_ms):.2f} ms per batch, FPS(combined): {fps_e2e_model_plus_io:.2f}')
        # 也可给出基于 sec_per_example 的端到端（含 dataloader）吞吐：
        logger.info(f'End-to-end sec_per_example: {sec_per_example*1000:.2f} ms per sample, FPS(e2e): {1.0/sec_per_example:.2f}')

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
