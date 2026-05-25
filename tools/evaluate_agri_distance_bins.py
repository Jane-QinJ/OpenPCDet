import argparse
import pickle
from pathlib import Path

import numpy as np


ROOT = Path('/home/firo/Documents/workspace/OpenPCDet')


EXPERIMENTS = {
    'farm_in_domain': {
        'result': ROOT / 'output/custom_models/pointpillar_farm_road/farm_road_in_domain/eval/eval_with_train/epoch_80/val/result.pkl',
        'info': Path('/home/firo/Documents/workspace/dataSet/L1401_kizu_T981_V420/custom_infos_val.pkl'),
    },
    'rice_in_domain': {
        'result': ROOT / 'output/custom_models/pointpillar_rice_field/rice_field_in_domain/eval/eval_with_train/epoch_80/val/result.pkl',
        'info': ROOT / 'data/custom/custom_infos_val.pkl',
    },
    'farm_to_rice': {
        'result': ROOT / 'output/custom_models/pointpillar_rice_field/farm_to_rice/eval/epoch_80/val/farm_to_rice/result.pkl',
        'info': ROOT / 'data/custom/custom_infos_val.pkl',
    },
    'rice_to_farm': {
        'result': ROOT / 'output/custom_models/pointpillar_farm_road/rice_to_farm/eval/epoch_80/val/rice_to_farm/result.pkl',
        'info': Path('/home/firo/Documents/workspace/dataSet/L1401_kizu_T981_V420/custom_infos_val.pkl'),
    },
    'mixed_to_farm': {
        'result': ROOT / 'output/custom_models/pointpillar_farm_road/mixed_to_farm/eval/epoch_80/val/mixed_to_farm/result.pkl',
        'info': Path('/home/firo/Documents/workspace/dataSet/L1401_kizu_T981_V420/custom_infos_val.pkl'),
    },
    'mixed_to_rice': {
        'result': ROOT / 'output/custom_models/pointpillar_rice_field/mixed_to_rice/eval/epoch_80/val/mixed_to_rice/result.pkl',
        'info': ROOT / 'data/custom/custom_infos_val.pkl',
    },
    'robust_mixed_to_farm': {
        'result': ROOT / 'output/custom_models/pointpillar_farm_road/robust_mixed_to_farm/eval/epoch_80/val/robust_mixed_to_farm/result.pkl',
        'info': Path('/home/firo/Documents/workspace/dataSet/L1401_kizu_T981_V420/custom_infos_val.pkl'),
    },
    'robust_mixed_to_rice': {
        'result': ROOT / 'output/custom_models/pointpillar_rice_field/robust_mixed_to_rice/eval/epoch_80/val/robust_mixed_to_rice/result.pkl',
        'info': ROOT / 'data/custom/custom_infos_val.pkl',
    },
}


BINS = [
    ('0-5m', 0.0, 5.0),
    ('5-10m', 5.0, 10.0),
    ('10-15m', 10.0, 15.0),
    ('15-20m', 15.0, 20.0),
    ('20-30m', 20.0, 30.0),
]


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def axis_aligned_iou_3d(boxes_a, boxes_b):
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    a_min = boxes_a[:, None, :3] - boxes_a[:, None, 3:6] / 2.0
    a_max = boxes_a[:, None, :3] + boxes_a[:, None, 3:6] / 2.0
    b_min = boxes_b[None, :, :3] - boxes_b[None, :, 3:6] / 2.0
    b_max = boxes_b[None, :, :3] + boxes_b[None, :, 3:6] / 2.0

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.clip(inter_max - inter_min, 0.0, None)
    inter_vol = inter[:, :, 0] * inter[:, :, 1] * inter[:, :, 2]

    vol_a = np.prod(boxes_a[:, 3:6], axis=1)[:, None]
    vol_b = np.prod(boxes_b[:, 3:6], axis=1)[None, :]
    return inter_vol / np.maximum(vol_a + vol_b - inter_vol, 1e-6)


def assign_matches(gt_boxes, pred_boxes, pred_scores, iou_thresh):
    gt_used = np.zeros(gt_boxes.shape[0], dtype=np.bool_)
    pred_used = np.zeros(pred_boxes.shape[0], dtype=np.bool_)
    pred_to_gt = np.full(pred_boxes.shape[0], -1, dtype=np.int64)
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return gt_used, pred_used, pred_to_gt

    order = np.argsort(-pred_scores)
    ious = axis_aligned_iou_3d(pred_boxes[order], gt_boxes)

    for pred_rank, pred_idx in enumerate(order):
        gt_idx = int(np.argmax(ious[pred_rank]))
        if ious[pred_rank, gt_idx] >= iou_thresh and not gt_used[gt_idx]:
            gt_used[gt_idx] = True
            pred_used[pred_idx] = True
            pred_to_gt[pred_idx] = gt_idx

    return gt_used, pred_used, pred_to_gt


def bin_name(distance):
    for name, low, high in BINS:
        if low <= distance < high:
            return name
    return '30m+'


def evaluate_experiment(name, result_path, info_path, iou_thresh=0.5, score_thresh=0.25):
    det_annos = load_pickle(result_path)
    infos = load_pickle(info_path)
    assert len(det_annos) == len(infos), (name, len(det_annos), len(infos))

    rows = {b[0]: {'gt': 0, 'tp': 0, 'pred': 0, 'fp': 0} for b in BINS}
    rows['30m+'] = {'gt': 0, 'tp': 0, 'pred': 0, 'fp': 0}

    for det, info in zip(det_annos, infos):
        gt_boxes = info['annos']['gt_boxes_lidar']
        pred_boxes = det['boxes_lidar']
        pred_scores = det['score']
        keep = pred_scores >= score_thresh
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        gt_matched, pred_matched, pred_to_gt = assign_matches(gt_boxes, pred_boxes, pred_scores, iou_thresh)

        for i, gt in enumerate(gt_boxes):
            b = bin_name(float(np.linalg.norm(gt[:2])))
            rows[b]['gt'] += 1

        for i, pred in enumerate(pred_boxes):
            if pred_matched[i]:
                gt_idx = pred_to_gt[i]
                b = bin_name(float(np.linalg.norm(gt_boxes[gt_idx, :2])))
                rows[b]['tp'] += 1
                rows[b]['pred'] += 1
            else:
                b = bin_name(float(np.linalg.norm(pred[:2])))
                rows[b]['fp'] += 1
                rows[b]['pred'] += 1

    out = []
    for b in [x[0] for x in BINS] + ['30m+']:
        r = rows[b]
        recall = r['tp'] / r['gt'] if r['gt'] else 0.0
        assert r['pred'] == r['tp'] + r['fp'], (name, b, r)
        precision = r['tp'] / r['pred'] if r['pred'] else 0.0
        out.append({
            'experiment': name,
            'distance_bin': b,
            'gt': r['gt'],
            'pred': r['pred'],
            'tp': r['tp'],
            'fp': r['fp'],
            'recall_iou50': recall,
            'precision_iou50': precision,
        })
    return out


def write_csv(rows, path):
    fields = ['experiment', 'distance_bin', 'gt', 'pred', 'tp', 'fp', 'recall_iou50', 'precision_iou50']
    lines = [','.join(fields)]
    for row in rows:
        lines.append(','.join(str(row[k]) if not isinstance(row[k], float) else '%.6f' % row[k] for k in fields))
    path.write_text('\n'.join(lines) + '\n')


def write_markdown(rows, path):
    lines = [
        '# Distance-binned Evaluation',
        '',
        'Metric uses greedy matching with axis-aligned 3D IoU >= 0.50 and score >= 0.25.',
        '',
        '| Experiment | Distance | GT | Pred | TP | FP | Recall | Precision |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            '| {experiment} | {distance_bin} | {gt} | {pred} | {tp} | {fp} | {recall_iou50:.3f} | {precision_iou50:.3f} |'.format(**row)
        )
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='/home/firo/Documents/workspace/OpenPCDet/output/agri_distance_eval')
    parser.add_argument('--include-missing', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, cfg in EXPERIMENTS.items():
        if not cfg['result'].exists():
            if args.include_missing:
                print('missing result:', name, cfg['result'])
            continue
        rows.extend(evaluate_experiment(name, cfg['result'], cfg['info']))

    write_csv(rows, out_dir / 'distance_binned_eval.csv')
    write_markdown(rows, out_dir / 'distance_binned_eval.md')
    print('Wrote', out_dir / 'distance_binned_eval.csv')
    print('Wrote', out_dir / 'distance_binned_eval.md')


if __name__ == '__main__':
    main()
