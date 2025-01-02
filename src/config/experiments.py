from config.templates import encoder, layers
from config.templates.experiment import Experiment
from config.templates.dataset import Dataset
from config.templates.models.simvc_comvc import CoMVC
from config.templates.models.custom import CAE, CAEKM, CAEKMLoss, CAELoss
from config.templates.models.dmsc import DMSC
from config.templates.models.eamc import EAMC
from config.templates.models.mimvc import MIMVC

# Default encoders for MNIST-like datasets
_MNIST_ENCODERS = [
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
    encoder.Encoder(input_size=(1, 28, 28), layers="cnn_small"),
]
_MNIST_DECODERS = [
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
    encoder.Encoder(input_size=None, layers="cnn_small_decoder"),
]

# ======================================================================================================================
# Blobs (debugging dataset)
# ======================================================================================================================
BLOBS_ENCODERS = [
    encoder.Encoder(layers="dense_2d", input_size=(2,)),
    encoder.Encoder(layers="dense_2d", input_size=(2,)),
]
BLOBS_DECODERS = [
    encoder.Encoder(layers="dense_2d_decoder", input_size=None),
    encoder.Encoder(layers="dense_2d_decoder", input_size=None),
]


class BlobsExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="blobs_dep")
    n_views: int = 2
    n_clusters: int = 3
    n_runs: int = 1
    n_epochs: int = 20


# ======================================================================================================================
# Noisy MNIST
# ======================================================================================================================
NOISY_MNIST_ENCODERS = _MNIST_ENCODERS
NOISY_MNIST_DECODERS = _MNIST_DECODERS


class NoisyMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="noisymnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Noisy FashionMNIST
# ======================================================================================================================
NOISY_FASHIONMNIST_ENCODERS = _MNIST_ENCODERS
NOISY_FASHIONMNIST_DECODERS = _MNIST_DECODERS


class NoisyFashionMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="noisyfashionmnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Edge MNIST
# ======================================================================================================================
EDGE_MNIST_ENCODERS = _MNIST_ENCODERS
EDGE_MNIST_DECODERS = _MNIST_DECODERS


class EdgeMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="edgemnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Edge FashionMNIST
# ======================================================================================================================
EDGE_FASHIONMNIST_ENCODERS = _MNIST_ENCODERS
EDGE_FASHIONMNIST_DECODERS = _MNIST_DECODERS


class EdgeFashionMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="edgefashionmnist")
    n_views: int = 2
    n_clusters: int = 10


# ======================================================================================================================
# Caltech101
# ======================================================================================================================

def _caltech_decoder(out_dim):
    return encoder.Encoder(layers=[
        layers.Dense(n_units=256),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=1024),
        layers.BatchNormalization(),
        layers.RELU,
        layers.Dense(n_units=out_dim, activation="sigmoid"),
    ])


CALTECH_ENCODERS = [
    encoder.Encoder(layers="dense_5", input_size=(48,)),
    encoder.Encoder(layers="dense_5", input_size=(40,)),
    encoder.Encoder(layers="dense_5", input_size=(254,)),
    encoder.Encoder(layers="dense_5", input_size=(1984,)),
    encoder.Encoder(layers="dense_5", input_size=(512,)),
    encoder.Encoder(layers="dense_5", input_size=(928,)),
]
CALTECH_DECODERS = [
    _caltech_decoder(48),
    _caltech_decoder(40),
    _caltech_decoder(254),
    _caltech_decoder(1984),
    _caltech_decoder(512),
    _caltech_decoder(928),
]


class Caltech7Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="caltech7")
    n_views: int = 6
    n_clusters: int = 7


class Caltech20Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="caltech20")
    n_views: int = 6
    n_clusters: int = 20


# ======================================================================================================================
# COIL-20
# ======================================================================================================================

COIL20_ENCODERS = [encoder.Encoder(input_size=(1, 64, 64), layers="cnn_large") for _ in range(3)]
COIL20_DECODERS = [encoder.Encoder(input_size=None, layers="cnn_large_decoder") for _ in range(3)]


class COIL20Experiment(Experiment):
    dataset_config: Dataset = Dataset(name="coil20")
    n_views: int = 3
    n_clusters: int = 20


# ======================================================================================================================
# PatchedMNIST
# ======================================================================================================================

PATCHED_MNIST_ENCODERS = _MNIST_ENCODERS
PATCHED_MNIST_DECODERS = _MNIST_DECODERS


class PatchedMNISTExperiment(Experiment):
    dataset_config: Dataset = Dataset(name="patchedmnist")
    n_views: int = 12
    n_clusters: int = 3


# ======================================================================================================================
# CoMVC
# ======================================================================================================================

blobs_comvc = BlobsExperiment(
    model_config=CoMVC(
        encoder_configs=BLOBS_ENCODERS
    ),
    batch_size=100,
)

noisymnist_comvc = NoisyMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

edgemnist_comvc = EdgeMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    ),
    batch_size=100,
)

caltech20_comvc = Caltech20Experiment(
    model_config=CoMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

caltech7_comvc = Caltech7Experiment(
    model_config=CoMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_comvc = NoisyFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

edgefashionmnist_comvc = EdgeFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

coil20_comvc = COIL20Experiment(
    model_config=CoMVC(
        encoder_configs=COIL20_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_comvc = PatchedMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)


# ======================================================================================================================
# CAE
# ======================================================================================================================

noisymnist_cae = NoisyMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
    )
)

edgemnist_cae = EdgeMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
    )
)

caltech20_cae = Caltech20Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

caltech7_cae = Caltech7Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

noisyfashionmnist_cae = NoisyFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
    )
)

edgefashionmnist_cae = EdgeFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
    )
)

coil20_cae = COIL20Experiment(
    model_config=CAE(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
    )
)

patchedmnist_cae = PatchedMNISTExperiment(
    model_config=CAE(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
    )
)


# ======================================================================================================================
# CAEKM
# ======================================================================================================================

noisymnist_caekm = NoisyMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
    )
)

edgemnist_caekm = EdgeMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
    )
)

caltech20_caekm = Caltech20Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

caltech7_caekm = Caltech7Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

noisyfashionmnist_caekm = NoisyFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
    )
)

edgefashionmnist_caekm = EdgeFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
    )
)

coil20_caekm = COIL20Experiment(
    model_config=CAEKM(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
    )
)

patchedmnist_caekm = PatchedMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
    )
)


# ======================================================================================================================
# SAE
# ======================================================================================================================

noisymnist_sae = NoisyMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgemnist_sae = EdgeMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech20_sae = Caltech20Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech7_sae = Caltech7Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

noisyfashionmnist_sae = NoisyFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgefashionmnist_sae = EdgeFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

coil20_sae = COIL20Experiment(
    model_config=CAE(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

patchedmnist_sae = PatchedMNISTExperiment(
    model_config=CAE(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        loss_config=CAELoss(funcs="DDC1|DDC2|DDC3|MSE"),
        pre_train_loss_config=CAELoss(funcs="MSE"),
        projector_config=None,
    )
)


# ======================================================================================================================
# SAEKM
# ======================================================================================================================

noisymnist_saekm = NoisyMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgemnist_saekm = EdgeMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech20_saekm = Caltech20Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech7_saekm = Caltech7Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

noisyfashionmnist_saekm = NoisyFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgefashionmnist_saekm = EdgeFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

coil20_saekm = COIL20Experiment(
    model_config=CAEKM(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

patchedmnist_saekm = PatchedMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE"),
        projector_config=None,
    )
)


# ======================================================================================================================
# DMSC
# ======================================================================================================================

DATASET_KWARGS = dict(
    pre_train_batch_size=100,
    pre_train_train_shuffle=True,
    pre_train_val_shuffle=True,
    train_shuffle=False,
    val_shuffle=False,
    test_shuffle=False,
)


blobs_dmsc = BlobsExperiment(
    dataset_config=Dataset(name="blobs_dep", **DATASET_KWARGS),
    model_config=DMSC(
        n_samples=3000,
        encoder_configs=BLOBS_ENCODERS,
        decoder_configs=BLOBS_DECODERS,
    ),
    batch_size=3000,
)


noisymnist_dmsc = NoisyMNISTExperiment(
    dataset_config=Dataset(name="noisymnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

edgemnist_dmsc = EdgeMNISTExperiment(
    dataset_config=Dataset(name="edgemnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

noisyfashionmnist_dmsc = NoisyFashionMNISTExperiment(
    dataset_config=Dataset(name="noisyfashionmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

edgefashionmnist_dmsc = EdgeFashionMNISTExperiment(
    dataset_config=Dataset(name="edgefashionmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

caltech20_dmsc = Caltech20Experiment(
    dataset_config=Dataset(name="caltech20", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        n_samples=2386,
    ),
    batch_size=2386
)

caltech7_dmsc = Caltech7Experiment(
    dataset_config=Dataset(name="caltech7", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        n_samples=1474,
    ),
    batch_size=1474
)

coil20_dmsc = COIL20Experiment(
    dataset_config=Dataset(name="coil20", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        n_samples=480,
    ),
    batch_size=480
)

patchedmnist_dmsc = COIL20Experiment(
    dataset_config=Dataset(name="patchedmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)


# ======================================================================================================================
# EAMC
# ======================================================================================================================

blobs_eamc = BlobsExperiment(
    model_config=EAMC(
        encoder_configs=BLOBS_ENCODERS,
    ),
    batch_size=100,
)

noisymnist_eamc = NoisyMNISTExperiment(
    model_config=EAMC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

edgemnist_eamc = EdgeMNISTExperiment(
    model_config=EAMC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    ),
    batch_size=100,
)

caltech20_eamc = Caltech20Experiment(
    model_config=EAMC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

caltech7_eamc = Caltech7Experiment(
    model_config=EAMC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_eamc = NoisyFashionMNISTExperiment(
    model_config=EAMC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

edgefashionmnist_eamc = EdgeFashionMNISTExperiment(
    model_config=EAMC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

coil20_eamc = COIL20Experiment(
    model_config=EAMC(
        encoder_configs=COIL20_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_eamc = PatchedMNISTExperiment(
    model_config=EAMC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)


# ======================================================================================================================
# MIMVC
# ======================================================================================================================

blobs_mimvc = BlobsExperiment(
    model_config=MIMVC(
        encoder_configs=BLOBS_ENCODERS
    )
)

noisymnist_mimvc = NoisyMNISTExperiment(
    model_config=MIMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    )
)

edgemnist_mimvc = EdgeMNISTExperiment(
    model_config=MIMVC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    )
)

caltech20_mimvc = Caltech20Experiment(
    model_config=MIMVC(
        encoder_configs=CALTECH_ENCODERS,
    )
)

caltech7_mimvc = Caltech7Experiment(
    model_config=MIMVC(
        encoder_configs=CALTECH_ENCODERS,
    )
)

noisyfashionmnist_mimvc = NoisyFashionMNISTExperiment(
    model_config=MIMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    )
)

edgefashionmnist_mimvc = EdgeFashionMNISTExperiment(
    model_config=MIMVC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    )
)

coil20_mimvc = COIL20Experiment(
    model_config=MIMVC(
        encoder_configs=COIL20_ENCODERS,
    )
)

patchedmnist_mimvc = PatchedMNISTExperiment(
    model_config=MIMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    )
)
