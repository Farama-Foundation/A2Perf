import pandas as pd

OPTIMAL_METRIC_CRITERIA = dict(
    returns='max',
    dispersion_across_runs='min',
    dispersion_within_runs='min',
    dispersion_across_rollouts='min',
    short_term_risk='min',
    long_term_risk='min',
    risk_across_runs='max',
    peak_ram_usage='min',
    ram_usage='min',
    wall_clock_time='min',
    inference_time='min'
    , gpu_power_usage='min'
)

METRIC_TO_DISPLAY_NAME = dict(
    returns='Returns',
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
    gpu_power_usage='GPU Power Usage'
)

METRIC_TO_CATEGORY = dict(
    returns='Application',
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
    gpu_power_usage='System'
)

METRIC_TO_UNIT = dict(
    returns='100 eps.',
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
    gpu_power_usage='W'

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
        display_val = f"{mean:.2f} ± {std:.2f}"
      else:
        value = values
        display_val = f"{value:.2f}"
      data_for_df.append(
          (domain, task, algo, category, display_name, unit, display_val))

  df = pd.DataFrame(data_for_df, columns=['domain', 'task', 'algo', 'category',
                                          'metric', 'unit', 'display_val'])
  return df


def df_as_latex(df, mode):
  # Merge 'metric' and 'unit' into one column, properly formatted
  df['metric'] = df.apply(lambda x: f"{x['metric']} ({x['unit']})", axis=1)

  # Drop the 'unit' column as it's no longer needed
  df = df.drop(columns='unit')

  # Reindex and sort the DataFrame
  df = df.set_index(
      ['domain', 'task', 'category', 'metric', 'algo']).sort_index()

  # Create a pivot table with 'category' and 'metric' as the row index, and 'algo' as the column index
  df_pivot = df.pivot_table(index=['category', 'metric'],
                            columns='algo',
                            values='display_val', aggfunc='first')

  df_pivot = df_pivot.rename_axis(
      index={'category': '\\textbf{Category}',
             'metric': '\\textbf{Metric Name}'})

  df_pivot.columns = [f'\\textbf{{{algo.upper()}}}' for algo in
                      df_pivot.columns]

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
      escape=False  # Allow LaTeX commands within cells
  )

  return latex_table
