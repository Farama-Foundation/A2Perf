import enum

import gin


@gin.constants_from_enum
class BenchmarkDomain(enum.Enum):
    QUADRUPED_LOCOMOTION = "QuadrupedLocomotion-v0"
    WEB_NAVIGATION = "WebNavigation-v0"
    CIRCUIT_TRAINING = "CircuitTraining-v0"


@gin.constants_from_enum
class BenchmarkMode(enum.Enum):
    TRAIN = "train"
    INFERENCE = "inference"
    GENERALIZATION = "generalization"


@gin.constants_from_enum
class SystemMetrics(enum.Enum):
    INFERENCE_TIME = "InferenceTime"
    TRAINING_TIME = "TrainingTime"
    MEMORY_USAGE = "MemoryUsage"


@gin.constants_from_enum
class ReliabilityMetrics(enum.Enum):
    IqrWithinRuns = "IqrWithinRuns"
    IqrAcrossRuns = "IqrAcrossRuns"
    LowerCVaROnDiffs = "LowerCVaROnDiffs"
    LowerCVaROnDrawdown = "LowerCVaROnDrawdown"
    LowerCVarOnAcross = "LowerCVarOnAcross"
    MedianPerfDuringTraining = "MedianPerfDuringTraining"
    MadAcrossRollouts = "MadAcrossRollouts"
    IqrAcrossRollouts = "IqrAcrossRollouts"
    StddevAcrossRollouts = "StddevAcrossRollouts"
    UpperCVaRAcrossRollouts = "UpperCVaRAcrossRollouts"
    LowerCVaRAcrossRollouts = "LowerCVaRAcrossRollouts"


ENV_NAMES = {
    BenchmarkDomain.QUADRUPED_LOCOMOTION: BenchmarkDomain.QUADRUPED_LOCOMOTION.value,
    BenchmarkDomain.WEB_NAVIGATION: BenchmarkDomain.WEB_NAVIGATION.value,
    BenchmarkDomain.CIRCUIT_TRAINING: BenchmarkDomain.CIRCUIT_TRAINING.value,
}
