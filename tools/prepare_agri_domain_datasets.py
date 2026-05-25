import argparse
from pathlib import Path


def read_ids(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def clear_link_dir(path):
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            raise RuntimeError('Refusing to remove non-file path: %s' % child)


def link_domain(domain, src_root, dst_root, split_name):
    split_ids = read_ids(src_root / 'ImageSets' / ('%s.txt' % split_name))
    out_ids = []
    for sample_id in split_ids:
        new_id = '%s_%s' % (domain, sample_id)
        for subdir, suffix in [('points', '.npy'), ('labels', '.txt')]:
            src = src_root / subdir / ('%s%s' % (sample_id, suffix))
            dst = dst_root / subdir / ('%s%s' % (new_id, suffix))
            if not src.exists():
                raise FileNotFoundError(src)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        out_ids.append(new_id)
    return out_ids


def main():
    parser = argparse.ArgumentParser(
        description='Create a mixed farm-road + rice-field custom dataset for OpenPCDet.'
    )
    parser.add_argument('--farm-root', default='/home/firo/Documents/workspace/dataSet/L1401_kizu_T981_V420')
    parser.add_argument('--rice-root', default='/home/firo/Documents/workspace/OpenPCDet/data/custom')
    parser.add_argument('--out-root', default='/home/firo/Documents/workspace/OpenPCDet/data/custom_mixed_farm_road_rice_field')
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Clear existing symlink/file entries in output points and labels first.',
    )
    args = parser.parse_args()

    farm_root = Path(args.farm_root).resolve()
    rice_root = Path(args.rice_root).resolve()
    out_root = Path(args.out_root).resolve()

    for root in [farm_root, rice_root]:
        for subdir in ['points', 'labels', 'ImageSets']:
            if not (root / subdir).exists():
                raise FileNotFoundError(root / subdir)

    ensure_dir(out_root / 'points')
    ensure_dir(out_root / 'labels')
    ensure_dir(out_root / 'ImageSets')
    if args.rebuild:
        clear_link_dir(out_root / 'points')
        clear_link_dir(out_root / 'labels')

    train_ids = []
    val_ids = []
    train_ids.extend(link_domain('farm', farm_root, out_root, 'train'))
    train_ids.extend(link_domain('rice', rice_root, out_root, 'train'))
    val_ids.extend(link_domain('farm', farm_root, out_root, 'val'))
    val_ids.extend(link_domain('rice', rice_root, out_root, 'val'))

    (out_root / 'ImageSets' / 'train.txt').write_text('\n'.join(train_ids) + '\n')
    (out_root / 'ImageSets' / 'val.txt').write_text('\n'.join(val_ids) + '\n')
    print('Wrote %d train and %d val samples to %s' % (len(train_ids), len(val_ids), out_root))


if __name__ == '__main__':
    main()
