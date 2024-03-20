import pandas as pd

OPTIMAL_METRIC_CRITERIA = dict(
    rollout_returns='max',
    dispersion_across_runs='min',
    dispersion_within_runs='min',
    dispersion_across_rollouts='min',
    short_term_risk='min',
    long_term_risk='min',
    risk_across_runs='max',
    peak_ram_usage='min',
    mean_ram_usage='min',
    wall_clock_time='min',
    inference_time='min',
    gpu_power_usage='min',
    risk_across_rollouts='max',
    disperion_across_rollouts='min',


)

METRIC_TO_DISPLAY_NAME = dict(
    rollout_returns='Returns',
    dispersion_across_runs='Dispersion Across Runs',
    dispersion_within_runs='Dispersion Within Runs',
    dispersion_across_rollouts='Dispersion Across Rollouts',
    short_term_risk='Short Term Risk',
    long_term_risk='Long Term Risk',
    risk_across_runs='Risk Across Runs',
    peak_ram_usage='Peak RAM Usage',
    mean_ram_usage='Mean RAM Usage',
    wall_clock_time='Wall Clock Time',
    inference_time='Inference Time',
    gpu_power_usage='GPU Power Usage',
    risk_across_rollouts='Risk Across Rollouts',
    disperion_across_rollouts='Dispersion Across Rollouts',
)

METRIC_TO_CATEGORY = dict(
    rollout_returns='Application',
    dispersion_across_runs='Reliability',
    dispersion_within_runs='Reliability',
    dispersion_across_rollouts='Reliability',
    short_term_risk='Reliability',
    long_term_risk='Reliability',
    risk_across_runs='Reliability',
    peak_ram_usage='System',
    mean_ram_usage='System',
    wall_clock_time='System',
    inference_time='System',
    gpu_power_usage='System',
    risk_across_rollouts='Reliability',
    disperion_across_rollouts='Reliability',
)

METRIC_TO_UNIT = dict(
    rollout_returns='100 eps.',
    dispersion_across_runs='IQR',
    dispersion_within_runs='IQR',
    dispersion_across_rollouts='IQR',
    short_term_risk='CVaR',
    long_term_risk='CVaR',
    risk_across_runs='CVaR',
    peak_ram_usage='GB',
    mean_ram_usage='GB',
    wall_clock_time='Hours',
    inference_time='ms',
    gpu_power_usage='W',
    risk_across_rollouts='CVaR',
    disperion_across_rollouts='IQR',
)


def metrics_dict_to_pandas_df(metrics_dict):
  # Transform the metrics dictionary into a DataFrame
  data_for_df = []
  for metric_name, metric_data in metrics_dict.items():
    for parameters, values in metric_data.items():
      domain, algo, task = parameters
      category = METRIC_TO_CATEGORY[metric_name]
      display_name = METRIC_TO_DISPLAY_NAME[metric_name]
      unit = METRIC_TO_UNIT[metric_name]
      if isinstance(values, dict):
        mean = values['mean']
        std = values['std']
        display_val = f'{mean:.2f} ± {std:.2f}'
      else:
        value_to_compare = values
        display_val = f'{value_to_compare:.2f}'
      data_for_df.append(
          (domain, task, algo, category, display_name, unit, display_val)
      )

  # For every metric, we must decide which algorithm is the best
  for metric_name, metric_data in metrics_dict.items():
    optimal_criterion = OPTIMAL_METRIC_CRITERIA[metric_name]

    best_exps = []
    best_value = None
    for (domain, algo, task), values in metric_data.items():
      category = METRIC_TO_CATEGORY[metric_name]
      display_name = METRIC_TO_DISPLAY_NAME[metric_name]
      unit = METRIC_TO_UNIT[metric_name]

      value_to_compare = values
      comparison_function = None

      if isinstance(value_to_compare, dict):
        value_to_compare = value_to_compare['mean']

      if best_value is None:
        comparison_function = lambda new, old: True
      elif optimal_criterion == 'min':
        comparison_function = lambda new, old: new < old
      elif optimal_criterion == 'max':
        comparison_function = lambda new, old: new > old

      if comparison_function is not None and comparison_function(
          value_to_compare, best_value):
        best_exps.clear()
        best_value = value_to_compare

      # Check for equality for the case where it's as good as the best_value (and best_value is not None)
      parameters_to_add =(domain, task, algo, category, display_name, unit, display_val)
      if best_value is not None and value_to_compare == best_value:
        best_exps.append(
            parameters_to_add)
      elif comparison_function(value_to_compare, best_value):
        best_exps = [
            parameters_to_add]
  # After comparing all experiments, for the optimal experiment we simply need to replace
  # the values with a latex bolded version
  for exp in best_exps:
    pass
  df = pd.DataFrame(
      data_for_df,
      columns=[
          'domain',
          'task',
          'algo',
          'category',
          'metric',
          'unit',
          'display_val',
      ],
  )
  return df


def df_as_latex(df, mode):
  # Merge 'metric' and 'unit' into one column, properly formatted
  df['metric'] = df.apply(lambda x: f"{x['metric']} ({x['unit']})", axis=1)

  # Drop the 'unit' column as it's no longer needed
  df = df.drop(columns='unit')

  # Reindex and sort the DataFrame
  df = df.set_index(
      ['domain', 'task', 'category', 'metric', 'algo']
  ).sort_index()

  # Create a pivot table with 'category' and 'metric' as the row index, and 'algo' as the column index
  df_pivot = df.pivot_table(
      index=['category', 'metric'],
      columns='algo',
      values='display_val',
      aggfunc='first',
  )

  df_pivot = df_pivot.rename_axis(
      index={
          'category': '\\textbf{Category}',
          'metric': '\\textbf{Metric Name}',
      }
  )

  df_pivot.columns = [
      f'\\textbf{{{algo.upper()}}}' for algo in df_pivot.columns
  ]

  # Determine the number of algorithms dynamically for column formatting
  column_format = '|l|l|' + 'c|' * len(df_pivot.columns)

  # Generate the LaTeX table
  latex_table = df_pivot.to_latex(
      index=True,
      multirow=True,
      multicolumn=True,
      position='!htbp',
      bold_rows=False,  # Set to False as we've already applied bold to headers
      column_format=column_format,
      caption='Metrics for the application domain',
      na_rep='N/A',
      label='tab:metrics',
      escape=False,  # Allow LaTeX commands within cells
  )

  return latex_table
