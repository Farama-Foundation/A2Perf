import gin

from rl_perf.metrics.profiler.eval_profiler import EvaluationProfiler
from rl_perf.metrics.profiler.training_profiler import TrainingProfiler


@gin.configurable
class TrainingMemoryProfiler(TrainingProfiler):
    def __init__(self):
        super(TrainingMemoryProfiler, self).__init__()


class EvaluationMemoryProfiler(EvaluationProfiler):
    def __init__(self):
        super(EvaluationMemoryProfiler, self).__init__()
