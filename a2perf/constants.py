import enum

import gin


@gin.constants_from_enum
class BenchmarkDomain(enum.Enum):
    QUADRUPED_LOCOMOTION = "quadruped_locomotion"
    WEB_NAVIGATION = "web_navigation"
    CIRCUIT_TRAINING = "circuit_training"


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
    BenchmarkDomain.QUADRUPED_LOCOMOTION: [
        "QuadrupedLocomotion-DogPace-v0",
        "QuadrupedLocomotion-DogTrot-v0",
        "QuadrupedLocomotion-DogSpin-v0",
    ],
    BenchmarkDomain.WEB_NAVIGATION: [
        "WebNavigation-Difficulty-01-v0",
        "WebNavigation-Difficulty-02-v0",
        "WebNavigation-Difficulty-03-v0",
    ],
    BenchmarkDomain.CIRCUIT_TRAINING: [
        "CircuitTraining-ToyMacro-v0",
        "CircuitTraining-Ariane-v0",
    ],
}
