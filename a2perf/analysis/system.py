def get_inference_time():

  # Mean/STD of the inference time metric
  inference_time_df = inference_df[inference_df['metric'] == 'inference_time']
  mean_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].mean()
  std_inference_time = inference_time_df.groupby(['domain', 'algo', 'task', ])[
    0].std()
  mean_inference_time, std_inference_time

  pass


def get_peak_ram_usage():
  training_peak_ram_usage = \
    training_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'ram_process'].max()

  training_mean_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).mean()
  training_std_peak_ram = training_peak_ram_usage.groupby(
      ['domain', 'algo', 'task']).std()

  pass


def get_mean_ram_usage():
  mean_training_ram_usage = \
    training_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].mean()
  std_training_ram_usage = \
    training_system_metrics_df.groupby(['domain', 'algo', 'task'])[
      'ram_process'].std()

  # Display the means and stds nicely
  for domain in mean_training_ram_usage.index.levels[0]:
    for algo in mean_training_ram_usage.index.levels[1]:
      for task in mean_training_ram_usage.index.levels[2]:
        all_training_metrics[(domain, algo, task)]['ram_usage'] = dict(
            mean=mean_training_ram_usage.loc[domain, algo, task],
            std=std_training_ram_usage.loc[domain, algo, task])

        print(
            f'Domain: {domain}, Algo: {algo}, Task: {task}, Mean RAM Usage: {mean_training_ram_usage.loc[domain, algo, task]}, Std RAM Usage: {std_training_ram_usage.loc[domain, algo, task]}')

  pass


def get_wall_clock_time():
  # In[ ]:

  # Get the last duration for each individual experiment
  last_durations = \
    training_system_metrics_df.groupby(
        ['domain', 'algo', 'task', 'experiment', 'seed'])[
      'duration'].last()

  # In[ ]:

  # Now find the mean and std of the last duration for each algo
  mean_last_duration = last_durations.groupby(['domain', 'algo', 'task']).mean()
  std_last_duration = last_durations.groupby(['domain', 'algo', 'task']).std()
  mean_last_duration

  # In[ ]:

  mean_last_duration.index.levels

  # In[ ]:

  # Display the means and stds nicely. Should be durations for domain/algo/task
  for domain in mean_last_duration.index.levels[0]:
    print(domain)
    for algo in mean_last_duration.index.levels[1]:
      print(f'\t{algo}')
      for task in mean_last_duration.loc[domain, algo].index:
        print(
            f'\t\tTask: {task}, Mean Last Duration: {mean_last_duration.loc[domain, algo, task]}, Std Last Duration: {std_last_duration.loc[domain, algo, task]}')
        all_training_metrics[(domain, algo, task)]['wall_clock_time'] = dict(
            mean=mean_last_duration.loc[domain, algo, task],
            std=std_last_duration.loc[domain, algo, task])


def get_mean_ram_usage():
  pass


def get_training_system_metrics():
  pass
