from config.experiments.base_experiments import *
from config.templates.models.mimvc import MIMVC


blobs_mimvc = BlobsExperiment(
    model_parameters=MIMVC(
        encoder_configs=BLOBS_ENCODERS
    )
)

noisymnist_mimvc = NoisyMNISTExperiment(
    model_parameters=MIMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    )
)

edgemnist_mimvc = EdgeMNISTExperiment(
    model_parameters=MIMVC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    )
)

caltech20_mimvc = Caltech20Experiment(
    model_parameters=MIMVC(
        encoder_configs=CALTECH_ENCODERS,
    )
)

caltech7_mimvc = Caltech7Experiment(
    model_parameters=MIMVC(
        encoder_configs=CALTECH_ENCODERS,
    )
)

noisyfashionmnist_mimvc = NoisyFashionMNISTExperiment(
    model_parameters=MIMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    )
)

edgefashionmnist_mimvc = EdgeFashionMNISTExperiment(
    model_parameters=MIMVC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    )
)

coil20_mimvc = COIL20Experiment(
    model_parameters=MIMVC(
        encoder_configs=COIL20_ENCODERS,
    )
)

patchedmnist_mimvc = PatchedMNISTExperiment(
    model_parameters=MIMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    )
)
