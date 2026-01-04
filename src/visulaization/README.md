To plot the reward comparison 

```bash
# multipanel
vis_baseline_reward.py --jobid 4675598 --multipanel --align left 

# individuals
vis_baseline_reward.py --jobid 4675598 --align left
```

To generate all plots at once (rewards, intrusions, crashes, etc.):

```bash
python generate_all_plots.py --jobid 4675598
```

This will run all visualization scripts with right alignment, creating both single and multipanel plots for rewards.