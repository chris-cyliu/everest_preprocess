import os
import json
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from run import parse_config, print_args, load_cm_result, calibrate_probability, build_uncertain_table, get_topk, evaluate


def reset_args(args, default_dict):
    args_dict = vars(args)
    for k, v in default_dict.items():
        args_dict[k] = v
        logging.info('Reset {}: {}'.format(k, v))


def modify_arg(args, key, value):
    args_dict = vars(args)
    args_dict[key] = value
    logging.info('Modify {} to {}'.format(key, value))


def modify_log_path(args):
    log_path = 'vdata/{}/logs/{}_k{}_conf{}_gap{}_calibrate{}.log'.format(
        'taipei-bus',
        'gt0.5_t0dt0.1',
        args.k,
        args.confidence,
        args.gap,
        args.calibrate
    )
    modify_arg(args, 'log_path', log_path)


def load_time(args):
    with open(args.time_dict_path, 'r') as f:
        time_dict = json.load(f)
        baseline = time_dict['baseline']['inference']
        cm_infer = time_dict['ours']['inference']

    infer_time = {'baseline': baseline, 'cm_infer': cm_infer}

    return infer_time


def run(args, scores, timestamp_list, image_path_list):
    elapsed = []

    t2 = 0
    if args.calibrate:
        t2, scores = calibrate_probability(args, scores)

    t3 = build_uncertain_table(args, scores, timestamp_list, image_path_list)
    t4, topk_value, topk_indices, niter, niter_select = get_topk(args)
    elapsed += [t2, t3, t4]
    prec, rec, rd, se = evaluate(args, topk_value, topk_indices, elapsed)

    return elapsed, niter, niter_select, [prec, rec, rd, se]


def plot_candidate(args,
                   axs,
                   default_dict,
                   candidate,
                   xlabels,
                   ylabels,
                   infer_time,
                   scores,
                   timestamp_list,
                   image_path_list,
                   colors=['red', 'blue']):
    reset_args(args, default_dict)

    num_subfig = len(axs)

    args_dict = vars(args)
    ret_dict = {
        'time': [],
        'speedup': [],
        'iter': [],
        'iter_select': [],
        'precision': [],
        'recall': [],
        'rank_distance': [],
        'score_error': []
    }
    for c in candidate:
        args_dict[xlabels[0]] = c
        modify_log_path(args)
        elapsed, niter, niter_select, metric = run(args, scores, timestamp_list, image_path_list)
        elapsed = [infer_time['cm_infer']] + elapsed
        ret_dict['time'].append(np.sum(elapsed))
        ret_dict['speedup'].append(np.sum(elapsed) / infer_time['baseline'])
        ret_dict['iter'].append(niter)
        ret_dict['iter_select'].append(np.mean(niter_select))
        ret_dict['precision'].append(metric[0])
        ret_dict['recall'].append(metric[1])
        ret_dict['rank_distance'].append(metric[2])
        ret_dict['score_error'].append(metric[3])

    for i in range(num_subfig):
        ax = axs[i]
        ylabel = ylabels[i]

        y0 = ret_dict[ylabel[0]]
        y1 = ret_dict[ylabel[1]]

        lns1 = ax.plot(candidate, y0, '.-', color=colors[0], label=ylabel[0])
        ax.set_xlabel(xlabels[0])
        ax.set_ylabel(ylabel[0])

        axt = ax.twinx()
        lns2 = axt.plot(candidate, y1, '.-', color=colors[1], label=ylabel[1])
        axt.set_ylabel(ylabel[1])

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='best')


if __name__ == '__main__':
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    args = parse_config()
    print_args(args)

    modify_arg(args, 'time_dict_path', 'vdata/taipei-bus/time.json')

    default_dict = {
        'k': args.k,
        'confidence': args.confidence,
        'calibrate': args.calibrate,
        'gap': args.gap
    }
    k_list = [1, 3, 5, 10, 20, 50, 100]
    conf_list = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
#    gap_list = [1, 3, 10, 30, 150, 300]

    os.chdir('./darknet')

    fig, ax = plt.subplots(nrows=4, ncols=4)
    plt.subplots_adjust(wspace=1.0, hspace=0.5)
    fig.set_size_inches(20, 10)

    infer_time = load_time(args)
    scores, timestamp_list, image_path_list = load_cm_result(args)

    plot_candidate(
        args,
        ax[0],
        default_dict,
        k_list,
        ['k'],
        [['time', 'speedup'], ['iter', 'iter_select'], ['precision', 'recall'], ['rank_distance', 'score_error']],
        infer_time,
        scores,
        timestamp_list,
        image_path_list
    )
    plot_candidate(
        args,
        ax[1],
        default_dict,
        conf_list,
        ['confidence'],
        [['time', 'speedup'], ['iter', 'iter_select'], ['precision', 'recall'], ['rank_distance', 'score_error']],
        infer_time,
        scores,
        timestamp_list,
        image_path_list
    )

    default_dict['calibrate'] = 1 - default_dict['calibrate']
    plot_candidate(
        args,
        ax[2],
        default_dict,
        k_list,
        ['k'],
        [['time', 'speedup'], ['iter', 'iter_select'], ['precision', 'recall'], ['rank_distance', 'score_error']],
        infer_time,
        scores,
        timestamp_list,
        image_path_list
    )
    plot_candidate(
        args,
        ax[3],
        default_dict,
        conf_list,
        ['confidence'],
        [['time', 'speedup'], ['iter', 'iter_select'], ['precision', 'recall'], ['rank_distance', 'score_error']],
        infer_time,
        scores,
        timestamp_list,
        image_path_list
    )
#    plot_candidate(
#        args,
#        ax[2],
#        default_dict,
#        gap_list,
#        ['gap'],
#        ['time(s)', 'iterations'],
#        infer_time
#    )

    fig.savefig('plot.pdf')
