def format_number(num):
  if -1000 < num < 1000:
    return f"{num:.2f}"
  else:
    return f"{num:.2e}"


def is_best(value, all_values, criteria):
  if criteria == 'max':
    return value == max(all_values)
  elif criteria == 'min':
    return value == min(all_values)
  return False


def get_metric_values(metrics_dict, domain, task, metric, criteria):
  metric_values = {}
  all_values = []

  # Collect all mean values for comparison
  for algo in ["PPO", "DDPG"]:
    key = (domain, algo, task)
    if key in metrics_dict and metric in metrics_dict[key]:
      values = metrics_dict[key][metric]
      mean_val = values['mean'] if isinstance(values, dict) else values
      all_values.append(mean_val)

  # Determine the best value and format numbers
  for algo in ["PPO", "DDPG"]:
    key = (domain, algo, task)
    if key in metrics_dict and metric in metrics_dict[key]:
      values = metrics_dict[key][metric]
      mean_val = values['mean'] if isinstance(values, dict) else values
      std_val = values['std'] if isinstance(values,
                                            dict) and 'std' in values else None

      is_bold = is_best(mean_val, all_values, criteria)

      # Special handling for specific metrics
      if metric == 'wall_clock_time':
        mean_val /= 3600
        std_val /= 3600
      elif metric == 'inference_time':
        mean_val *= 1000
        std_val *= 1000

      mean_str = format_number_bold(mean_val, is_bold)
      if std_val:
        std_str = format_number_bold(std_val,
                                     is_bold)  # Bold std if mean is bold
        metric_values[algo] = f"{mean_str} $\\pm$ {std_str}"
      else:
        metric_values[algo] = f"{mean_str}"

    else:
      metric_values[algo] = "N/A"

  return metric_values


def add_metric_rows_to_table(metrics_table, metrics_dict, domain, task,
    criteria_dict):
  latex_table_rows = []
  for category, metrics in metrics_table['category'].items():
    units = metrics_table['units']
    unit_list = units[category]
    first_metric = True
    for i, metric in enumerate(metrics):
      criteria = criteria_dict.get(metric,
                                   'max')  # Default to 'max' if not specified
      metric_values = get_metric_values(metrics_dict, domain, task, metric,
                                        criteria)
      metric_name = f'{metric.replace("_", " ").title()} ({unit_list[i]})'
      row = f'\t\t & {metric_name} & {metric_values["PPO"]} & {metric_values["DDPG"]} \\\\\n'
      if first_metric:
        row = f'\t\t\\multirow{{{len(metrics)}}}{{*}}{{{category}}}' + row
        first_metric = False
      latex_table_rows.append(row)
    latex_table_rows.append('\t\t\\hline\n')
  return ''.join(latex_table_rows)


def format_number_bold(num, is_bold):
  formatted_num = format_number(num)
  return f"\\textbf{{{formatted_num}}}" if is_bold else formatted_num


def generate_latex_table_header(task_name, phase):
  header_title = task_name.replace('_', ' ').title()
  header = [
      f'\\multicolumn{{4}}{{|c|}}{{\\textbf{{{header_title} ({phase})}}}} \\\\',
      '\\hline',
      '\\textbf{Category} & \\textbf{Metric Name} & \\textbf{PPO} & \\textbf{DDPG} \\\\',
      '\\hline'
  ]
  return '\n'.join(header)


def build_results_table():
  for domain, domain_group in all_df.groupby('domain'):
    for task, task_group in domain_group.groupby('task'):
      header_title = task.replace('_', ' ').title()
      latex_table_string = ['\\begin{figure}[!htbp]',
                            '\\resizebox{1\\textwidth}{!}{',
                            '\\begin{tabular}{|c|l|c|c|}',
                            '\\hline', ]
      latex_table_string.append(generate_latex_table_header(task, "Training"))

      # Add training metric rows with criteria
      latex_table_string.append(
          add_metric_rows_to_table(train_table, all_training_metrics, domain,
                                   task, criteria_dict))

      # Add inference header
      latex_table_string.append(generate_latex_table_header(task, "Inference"))

      # Add inference metric rows with criteria
      latex_table_string.append(
          add_metric_rows_to_table(inference_table, all_inference_metrics,
                                   domain,
                                   task, criteria_dict))

      # Closing tags
      latex_table_string.extend(
          ['\t\\end{tabular}}\n',
           f'\t\\caption{{Training and Inference Metrics for {header_title}}}\n',
           f'\t\\label{{tab:appendix_metrics_{task}}}\n', '\\end{figure}\n'])

      latex_table = ''.join(latex_table_string)
      print(latex_table)
      print("\n\n")
  pass


def display_results():
  train_table = dict(
      category={
          "Application": ["returns"],
          "Reliability": ["dispersion_across_runs", "dispersion_within_runs",
                          "short_term_risk",
                          "long_term_risk", "risk_across_runs"],
          "System": ["peak_ram_usage", "ram_usage", "wall_clock_time"]
      }
      ,
      units={
          "Application": ['100 eps.'],
          "Reliability": ["IQR", "IQR", "CVaR", "CVaR", "CVaR"],
          "System": ["GB", "GB", "Hours"]
      })

  inference_table = dict(
      category={
          "Application": ['N/A'],
          "Reliability": ["dispersion_across_rollouts", "risk_across_rollouts"],
          "System": ["peak_ram_usage", "ram_usage", "inference_time"]
      },
      units={
          "Application": [''],
          "Reliability": ["IQR", "CVaR"],
          "System": ["GB", "GB", "ms"]
      }
  )

  criteria_dict = dict(
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
  )
