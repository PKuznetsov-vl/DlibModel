import argparse
import json
import os
import os.path
from collections import defaultdict
from utils.file_managment import FileOp
import numpy as np
from numpy import trapz
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_points(dir_path, max_points, point_f):
    print('Reading directory {}'.format(dir_path))
    points = {}

    for idx, fname in enumerate(os.listdir(dir_path)):
        if fname in point_f:
            if max_points is not None and idx > max_points:
                break

            cur_path = os.path.join(dir_path, fname)

            if not os.path.exists(dir_path):
                print('Wrong directory')
                break

            if cur_path.endswith('.pts') or cur_path.endswith('.pts1'):
                if idx % 100 == 0:
                    print(idx)

                with open(cur_path) as cur_file:
                    lines = cur_file.readlines()
                    if lines[0].startswith('version'):  # to support different formats
                        lines = lines[3:-1]
                    mat = np.fromstring(''.join(lines), sep=' ')
                    points[fname] = (mat[0::2], mat[1::2])
        else:
            print('Not in Names')

    return points


def count_ced(predicted_points, gt_points, args):
    ceds = defaultdict(list)
    normalization_type = 'bbox'
    for method_name in predicted_points.keys():
        print('Counting ces. Method name {}'.format(method_name))
        for img_name in predicted_points[method_name].keys():
            if img_name in gt_points:
                # print('Processing key {}'.format(img_name))
                x_pred, y_pred = predicted_points[method_name][img_name]
                x_gt, y_gt = gt_points[img_name]
                n_points = x_pred.shape[0]
                assert n_points == x_gt.shape[0], '{} != {}'.format(n_points, x_gt.shape[0])

                if normalization_type == 'eyes':
                    left_eye_idx = args.left_eye_idx.split(',')
                    right_eye_idx = args.right_eye_idx.split(',')
                    if (len(left_eye_idx) == 1 and len(right_eye_idx) == 1) or \
                            (len(left_eye_idx) == 2 and len(right_eye_idx) == 2):
                        x_left_eye = np.mean([x_gt[int(idx)] for idx in left_eye_idx])
                        x_right_eye = np.mean([x_gt[int(idx)] for idx in right_eye_idx])
                        y_left_eye = np.mean([y_gt[int(idx)] for idx in left_eye_idx])
                        y_right_eye = np.mean([y_gt[int(idx)] for idx in right_eye_idx])
                    else:
                        raise Exception("Wrong number of eye points")

                    normalization_factor = np.linalg.norm([x_left_eye - x_right_eye, y_left_eye - y_right_eye])
                elif normalization_type == 'bbox':
                    w = np.max(x_gt) - np.min(x_gt)
                    h = np.max(y_gt) - np.min(y_gt)
                    normalization_factor = np.sqrt(h * w)
                else:
                    raise Exception('Wrong normalization type')

                diff_x = [x_gt[i] - x_pred[i] for i in range(n_points)]
                diff_y = [y_gt[i] - y_pred[i] for i in range(n_points)]
                dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
                avg_norm_dist = np.sum(dist) / (n_points * normalization_factor)
                ceds[method_name].append(avg_norm_dist)
                # print('Average distance for method {} = {}'.format(method_name, avg_norm_dist))
            else:
                print('Skipping key {}, because its not in the gt points'.format(img_name))
        ceds[method_name] = np.sort(ceds[method_name])

    return ceds


def count_ced_auc(errors):
    if not isinstance(errors, list):
        errors = [errors]

    aucs = []
    for error in errors:
        auc = 0
        proportions = np.arange(error.shape[0], dtype=np.float32) / error.shape[0]
        assert (len(proportions) > 0)

        step = 0.01
        for thr in np.arange(0.0, 1.0, step):
            gt_indexes = [idx for idx, e in enumerate(error) if e >= thr]
            if len(gt_indexes) > 0:
                first_gt_idx = gt_indexes[0]
            else:
                first_gt_idx = len(error) - 1
            auc += proportions[first_gt_idx] * step
        aucs.append(auc)
    return aucs


def main():
    with open('config_plot.json') as file:
        config = json.load(file)
    file.close()
    gt_path = config['orig_data_path']
    predictions_path = config['predictor_data_path']
    output_path = config['graph_output_path']
    max_points_to_read = 68
    error_thr = 0.08
    choices = ['bbox', 'eyes']

    parser = argparse.ArgumentParser(description='CED computation script',
                                     add_help=True)
    parser.add_argument('--left_eye_idx', action='store', type=str, help='')
    parser.add_argument('--right_eye_idx', action='store', type=str, help='')
    args = parser.parse_args()

    print('error threshold = {}'.format(error_thr))
    print(predictions_path)

    files = FileOp(predictions_path)
    im, pts, fullpath = files.get_files_from_dir()

    predicted_points = {}
    # pred_path in args.predictions_path:
    predicted_points[os.path.basename(predictions_path)] = read_points(predictions_path, max_points_to_read, pts)

    gt_points = read_points(gt_path, max_points_to_read, pts)
    # print(predicted_points.keys())
    # print(gt_points)

    ceds = count_ced(predicted_points, gt_points, args)

    # saving figure
    line_styles = [':', '-.', '--', '-']
    plt.figure(figsize=(30, 20), dpi=100)
    for method_idx, method_name in enumerate(ceds.keys()):
        print('Plotting graph for the method {}'.format(method_name))
        err = ceds[method_name]
        proportion = np.arange(err.shape[0], dtype=np.float32) / err.shape[0]
        under_thr = err > error_thr
        last_idx = len(err)
        if len(np.flatnonzero(under_thr)) > 0:
            last_idx = np.flatnonzero(under_thr)[0]
        under_thr_range = range(last_idx)
        cur_auc = count_ced_auc(err)[0]

        area = trapz(err[under_thr_range], proportion[under_thr_range])
        print(area)
        plt.plot(err[under_thr_range], proportion[under_thr_range], label=method_name +
                                                                          ', auc={:1.3f},area={}'.format(cur_auc, area),
                 linestyle=line_styles[method_idx % len(line_styles)], linewidth=2.0)
    plt.legend(loc='right', prop={'size': 24})
    plt.savefig(output_path)


main()
