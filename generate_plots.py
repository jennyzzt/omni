import argparse
import numpy as np
import os
import pandas as pd
import re
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu

from crafter import constants
from envs.env_utils import get_synonym_tasks, get_repeat_tasks, get_compound_tasks


def tflogs2csv(args):
    dataframe = pd.DataFrame({'run': [], 'metric': [], 'value': [], 'smvalue': [], 'step': []})
    log_folders = [f for f in os.scandir(args.path) if f.is_dir()]
    csv_path = os.path.join(args.path, 'events_log.csv')
    fnum = 1
    for folder in log_folders:
        folder_path = folder.path
        folder_name = folder.name
        # if file is not dir, continue
        if not os.path.isdir(folder_path):
            continue
        print(f'Reading from {folder_name}')
        # loading tensorboard logs to pandas dataframe
        event_acc = EventAccumulator(folder_path)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x : x.value, event_list))
            steps = list(map(lambda x : x.step, event_list))
            # smooth data
            smooth_values = gaussian_filter1d(values, sigma=1.0, mode='nearest')
            run_name = folder_name
            r = {
                'run': [run_name] * len(steps),
                'metric': [tag] * len(steps),
                'value': values,
                'smvalue': smooth_values,
                'step': steps,
            }
            dataframe = pd.DataFrame(r)
            dataframe.to_csv(csv_path, mode='w' if fnum == 1 else 'a', index=False, header=True if fnum == 1 else False)
            fnum += 1
    # saving dataframe as csv
    print(f'Saved to {csv_path}')


def bootstrap_ci(series, n_boot=1000, ci=95, seed=None):
    """Bootstrap Confidence Interval using median."""
    np.random.seed(seed)
    boot_medians = np.array([series.sample(n=len(series), replace=True).median() for _ in range(n_boot)])
    lower = np.percentile(boot_medians, (100-ci)/2)
    upper = np.percentile(boot_medians, ci + (100-ci)/2)
    return lower, upper


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='./storage', help="path to logs path")
    parser.add_argument("--svg", action="store_true", default=False, help="save image files as svg")
    parser.add_argument("--skip-csv", action="store_true", default=False,
                        help="skip the generation of csv files from tensorflow logs")
    parser.add_argument("--trunc-steps", default=0, type=int, help="truncate the number of steps plotted")
    parser.add_argument("--smooth", action="store_true", default=False, help="gaussian smooth the values")
    # toggle type of tasks plotted
    parser.add_argument("--add-syn", action="store_true", default=False, help="plot synonym tasks too")
    parser.add_argument("--add-comp", action="store_true", default=False, help="plot compound tasks too")
    parser.add_argument("--int-only", action="store_true", default=False, help="plot interesting tasks only")
    parser.add_argument("--no-dummy", action="store_true", default=False, help="do not add dummy task")
    # toggle plot visualizations
    parser.add_argument("--no-legend", action="store_true", default=False,
                        help="do not plot the legend")
    parser.add_argument("--plt-context", default='notebook',
                        help="seaborn plot context, other options: poster, paper, talk")
    parser.add_argument("--long-hm", action="store_true", default=False, help="longer plots for heatmaps")
    parser.add_argument("--no-title", action="store_true", default=False, help="no title on plots")
    parser.add_argument("--no-axlabel", action="store_true", default=False, help="no axis labels on plots")
    parser.add_argument("--label-end", action="store_true", default=False, help="label the last datapoint")
    parser.add_argument("--no-ticklabel", action="store_true", default=False, help="no tick labels")
    parser.add_argument("--no-cbar", action="store_true", default=False, help="no colorbar")
    parser.add_argument("--transp-bg", action="store_true", default=False, help="set plot background to be transparent")
    args = parser.parse_args()

    sns.set_context(context=args.plt_context)
    filetype = 'png' if not args.svg else 'svg'

    # save tensorflow logs as csv files
    if not args.skip_csv:
        tflogs2csv(args)

    # generate folder to store plots
    plots_dir = os.path.join(args.path, 'plots/')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # load csv file
    csv_path = os.path.join(args.path, 'events_log.csv')
    dataframe = pd.read_csv(csv_path)
    if args.trunc_steps > 0:  # truncate plots to determined timestep
        dataframe = dataframe.drop(dataframe[dataframe['step'] > args.trunc_steps + 1].index)
    dataframe_og = dataframe.copy()
    dataframe.run = ['_'.join(rn.split('-')[:-1]) for rn in dataframe.run]

    # ordered tasks based on tech tree
    ordered_tasks = []
    all_achreqs = constants.achievements_requisites
    while len(ordered_tasks) != len(all_achreqs):
        for ach, reqs in all_achreqs.items():
            if ach not in ordered_tasks and all(r in ordered_tasks for r in reqs):
                ordered_tasks.append(ach)
    if not args.int_only:
        ordered_tasks, _ = get_repeat_tasks(ordered_tasks, counts=10)
        # ordered_tasks = [x for x in ordered_tasks if any([x.isdigit() for x in x.split('_')])]
        if args.add_comp:
            ordered_tasks += ['__and__'.join(xs) for xs in get_compound_tasks(constants.achievements, maxcomp=2, naive=False)]
    if args.add_syn:
        ordered_tasks, _ = get_synonym_tasks(ordered_tasks)
    if not args.no_dummy:
        ordered_tasks.append('dummy')

    # heatmap plots
    unique_runs = dataframe.run.unique()
    for runname in unique_runs:
        dftask = dataframe[dataframe.run == runname].copy()
        dftask = dftask[dftask.metric.str.contains('train_eval/.*-sr', regex=True)]
        dftask.loc[:, 'metric'] = list(map(lambda tn: re.search('train_eval/(.*)-sr', tn).group(1), dftask.metric))
        dftask.metric = dftask.metric.astype('category')
        dftask.metric = dftask.metric.cat.set_categories(ordered_tasks)
        dftask.sort_values(['metric'])
        dftask['step'] = dftask['step'].div(1e3)
        dftask['value'] = [max(v, 0.00001) for v in dftask['value']]  # ensure that 0 values are not white
        dftask = dftask.pivot_table(index='metric', columns='step', values='value', aggfunc='mean')
        if args.long_hm:
            pfig = plt.figure(figsize=(16, 30))
            # pfig = plt.figure(figsize=(16, 50))
            # pfig = plt.figure(figsize=(16, 120))
        else:
            pfig = plt.figure(figsize=(16, 15))
        # sns.set(font_scale=3)
        fig = sns.heatmap(dftask, norm=LogNorm(vmin=0.001, vmax=1.0), cbar=not args.no_cbar)
        # fig = sns.heatmap(dftask, vmin=0.0, vmax=1.0)
        if not args.no_title:
            fig.set_title('task success rates')
        if args.no_axlabel:
            fig.set(xlabel=None)
            fig.set(ylabel=None)
        if args.no_ticklabel:
            plt.xticks(ticks=plt.xticks()[0], labels=[''] * len(plt.xticks()[0]))
            plt.yticks(ticks=plt.yticks()[0], labels=[''] * len(plt.yticks()[0]))
            # labels_y = [item.get_text() for item in fig.get_yticklabels()]
            # labels_y = [label.replace('__', ' ').replace('_', ' ') for label in labels_y]
            # fig.set_yticklabels(labels_y)
        if args.transp_bg:
            pfig.set_alpha(0)
            plt.gca().patch.set_alpha(0)
        plt.savefig(os.path.join(plots_dir, f'{runname}-task_sr.{filetype}'),
                    bbox_inches="tight", transparent=args.transp_bg)
        plt.close()
        print(f'TSR Plot for {runname} saved to {plots_dir}')

        dftask = dataframe[dataframe.run == runname].copy()
        dftask = dftask[dftask.metric.str.contains('train_sampled/.*', regex=True)]
        dftask.loc[:, 'metric'] = list(map(lambda tn: re.search('train_sampled/(.*)', tn).group(1), dftask.metric))
        dftask.metric = dftask.metric.astype('category')
        dftask.metric = dftask.metric.cat.set_categories(ordered_tasks)
        dftask.sort_values(['metric'])
        dftask['step'] = dftask['step'].div(1e3)
        dftask['value'] = [max(v, 0.00001) for v in dftask['value']]  # ensure that 0 values are not white
        dftask = dftask.pivot_table(index='metric', columns='step', values='value', aggfunc='mean')
        if args.long_hm:
            pfig = plt.figure(figsize=(16, 30))
            # pfig = plt.figure(figsize=(16, 50))
            # pfig = plt.figure(figsize=(16, 120))
        else:
            plt.figure(figsize=(16, 15))
        fig = sns.heatmap(dftask, norm=LogNorm(vmin=0.001, vmax=1.0), cbar=not args.no_cbar)
        # fig = sns.heatmap(dftask, vmin=0.0, vmax=0.5)
        if not args.no_title:
            fig.set_title('task sampled rates')
        if args.no_axlabel:
            fig.set(xlabel=None)
            fig.set(ylabel=None)
        if args.no_ticklabel:
            plt.xticks(ticks=plt.xticks()[0], labels=[''] * len(plt.xticks()[0]))
            plt.yticks(ticks=plt.yticks()[0], labels=[''] * len(plt.yticks()[0]))
        if args.transp_bg:
            pfig.set_alpha(0)
            plt.gca().patch.set_alpha(0)
        plt.savefig(os.path.join(plots_dir, f'{runname}-task_sar.{filetype}'),
                    bbox_inches="tight", transparent=args.transp_bg)
        plt.close()
        print(f'TSaR Plot for {runname} saved to {plots_dir}')

    # plot for avg task success rate
    unique_runs = dataframe_og.run.unique()
    plt.figure(figsize=(16, 12))
    hue_order = None  # set the hue order here if needed
    newdfs = []
    for runname in unique_runs:
        dfstat = dataframe_og[dataframe_og.run == runname].copy()
        dfstat = dfstat[dfstat.metric.str.contains('train_eval/.*-sr', regex=True)]
        dfstat.loc[:, 'metric'] = list(map(lambda tn: re.search('train_eval/(.*)-sr', tn).group(1), dfstat.metric))
        dfstat = dfstat[dfstat.metric.isin(ordered_tasks)]
        unique_steps = dfstat.step.unique()
        vals = []
        for ustep in unique_steps:
            tsr_vals = dfstat[dfstat.step == ustep]
            tsr_vals = tsr_vals.smvalue if args.smooth else tsr_vals.value
            vals.append(tsr_vals.sum() / len(tsr_vals))
        newdf = pd.DataFrame({
            'run': ['_'.join(runname.split('-')[:-1])] * len(vals),
            'metric': ['avg_tsr'] * len(vals),
            'value': vals,
            'step': unique_steps,
        })
        newdfs.append(newdf)
    newdfstat = pd.concat(newdfs)

    # # significance testing
    # timestep_percents = [0.25, 0.5, 0.75, 1.0]
    # all_timesteps = sorted(newdfstat.step.unique())
    # unique_runs = newdfstat.run.unique()
    # for tsp in timestep_percents:
    #     uts_index = int(tsp * len(all_timesteps)) - 1
    #     lts_index = uts_index - 1
    #     utarget_timestep = all_timesteps[uts_index]
    #     ltarget_timestep = all_timesteps[lts_index]
    #     data = newdfstat.loc[newdfstat['step'] <= utarget_timestep]
    #     data = data.loc[data['step'] > ltarget_timestep]
    #     group_a = data.loc[data['run'] == unique_runs[0]].value
    #     group_b = data.loc[data['run'] == unique_runs[1]].value
    #     u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
    #     print(f'timestep percent: {tsp}, u_stat: {u_stat}, p_value: {p_value}')

    fig = sns.lineplot(
        data=newdfstat, x='step', y='value', hue='run', hue_order=hue_order,
        err_style='band', errorbar=('ci', 95), estimator='median',
        n_boot=1000, legend=(False if args.no_legend else 'auto'),
    )
    if args.label_end:
        for l in fig.lines:
            y = l.get_ydata()
            if len(y)>0:
                fig.annotate(f'{y[-1]:.4f}', xy=(1,y[-1]), xycoords=('axes fraction', 'data'),
                             ha='left', va='center', color=l.get_color())
        last_step_per_run = newdfstat.groupby('run')['step'].max()
        last_step_df = newdfstat[newdfstat.set_index(['run', 'step']).index.isin(last_step_per_run.items())]
        medians = last_step_df.groupby('run')['value'].apply(np.median).round(4)
        confidence_intervals = last_step_df.groupby('run')['value'].apply(bootstrap_ci)
        confidence_intervals = confidence_intervals.apply(lambda x: (round(x[0], 4), round(x[1], 4)))
        tmpdf = pd.concat([medians, confidence_intervals], axis=1)
        fig.annotate(f'{tmpdf.to_string()}', xy=(0.6,0.3), xycoords='axes fraction')
    if not args.no_title:
        fig.set_title('average task success rate')
    if args.no_axlabel:
        fig.set(xlabel=None)
        fig.set(ylabel=None)
    if args.transp_bg:
        pfig.set_alpha(0)
        plt.gca().patch.set_alpha(0)
    plt.savefig(os.path.join(plots_dir, f'avg_tsr.{filetype}'),
                bbox_inches="tight", transparent=args.transp_bg)
    plt.close()
    print(f'avg-TSR plot saved to {plots_dir}')

    # plot for craftscore
    thresholds = [0.05, 0.1, 0.2, 0.4]
    # thresholds = [0.2]
    for threshold in thresholds:
        unique_runs = dataframe_og.run.unique()
        plt.figure(figsize=(16, 12))
        hue_order = None  # set the hue order here if needed
        newdfs = []
        for runname in unique_runs:
            dfstat = dataframe_og[dataframe_og.run == runname].copy()
            dfstat = dfstat[dfstat.metric.str.contains('train_eval/.*-sr', regex=True)]
            dfstat.loc[:, 'metric'] = list(map(lambda tn: re.search('train_eval/(.*)-sr', tn).group(1), dfstat.metric))
            unique_steps = dfstat.step.unique()
            vals = []
            for ustep in unique_steps:
                tsr_data = dfstat[dfstat.step == ustep]
                tsr_vals = []
                tsr_metric = []
                for _, dp in tsr_data.iterrows():
                    if dp.metric not in ordered_tasks:
                        continue
                    elif dp.metric not in tsr_metric:
                        tsr_metric.append(dp.metric)
                        tsr_vals.append(dp.smvalue if args.smooth else dp.value)
                vals.append(sum([v > threshold for v in tsr_vals]))
            newdf = pd.DataFrame({
                'run': ['_'.join(runname.split('-')[:-1])] * len(vals),
                'metric': ['craftscore'] * len(vals),
                'value': vals,
                'step': unique_steps,
            })
            newdfs.append(newdf)
        newdfstat = pd.concat(newdfs)

        # # significance testing
        # timestep_percents = [0.25, 0.5, 0.75, 1.0]
        # all_timesteps = sorted(newdfstat.step.unique())
        # unique_runs = newdfstat.run.unique()
        # for tsp in timestep_percents:
        #     uts_index = int(tsp * len(all_timesteps)) - 1
        #     lts_index = uts_index - 1
        #     utarget_timestep = all_timesteps[uts_index]
        #     ltarget_timestep = all_timesteps[lts_index]
        #     data = newdfstat.loc[newdfstat['step'] <= utarget_timestep]
        #     data = data.loc[data['step'] > ltarget_timestep]
        #     group_a = data.loc[data['run'] == unique_runs[0]].value
        #     group_b = data.loc[data['run'] == unique_runs[1]].value
        #     u_stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')
        #     print(f'timestep percent: {tsp}, u_stat: {u_stat}, p_value: {p_value}')

        fig = sns.lineplot(
            data=newdfstat, x='step', y='value', hue='run', hue_order=hue_order,
            err_style='band', errorbar=('ci', 95), estimator='median',
            n_boot=1000, legend=(False if args.no_legend else 'auto'),
        )
        if args.label_end:
            for l in fig.lines:
                y = l.get_ydata()
                if len(y)>0:
                    fig.annotate(f'{y[-1]:.2f}', xy=(1,y[-1]), xycoords=('axes fraction', 'data'),
                                 ha='left', va='center', color=l.get_color())
            last_step_per_run = newdfstat.groupby('run')['step'].max()
            last_step_df = newdfstat[newdfstat.set_index(['run', 'step']).index.isin(last_step_per_run.items())]
            medians = last_step_df.groupby('run')['value'].apply(np.median).round(4)
            confidence_intervals = last_step_df.groupby('run')['value'].apply(bootstrap_ci)
            confidence_intervals = confidence_intervals.apply(lambda x: (round(x[0], 4), round(x[1], 4)))
            tmpdf = pd.concat([medians, confidence_intervals], axis=1)
            fig.annotate(f'{tmpdf.to_string()}', xy=(0.6,0.3), xycoords='axes fraction')
        if not args.no_title:
            fig.set_title(f'tasks done with >{threshold} success rate')
        if args.no_axlabel:
            fig.set(xlabel=None)
            fig.set(ylabel=None)
        if args.transp_bg:
            pfig.set_alpha(0)
            plt.gca().patch.set_alpha(0)
        plt.savefig(os.path.join(plots_dir, f'craftscore{threshold}.{filetype}'),
                    bbox_inches="tight", transparent=args.transp_bg)
        plt.close()
        print(f'Craftscore plot saved to {plots_dir}')
