import torch as th
from torch import nn

from models.base import BaseModelKMeans, BaseModelPreTrain
from models.base.base_model_spectral import BaseModelSpectral
from models.clustering_module import get_clustering_module
from models.clustering_module import kernel_width
from lib.encoder import Encoder, EncoderList
from lib.fusion import get_fusion_module
from lib.normalization import get_normalizer
from register import register_model


@register_model
class CoMVC(SiMVC):
    def __init__(self, cfg):
        super(CoMVC, self).__init__(cfg)

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = Encoder(cfg.projector_config)

        self.projections = None

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.projections = [self.projector(x) for x in self.encoder_outputs]
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output


@register_model
class CAE(BaseModelPreTrain):
    def __init__(self, cfg):
        super(CAE, self).__init__(cfg)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes_before_flatten)

        self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=self.fusion.output_size)

        if cfg.projector_config is not None:
            self.projector = Encoder(cfg.projector_config)
        else:
            self.projector = nn.Identity()

        self.views = None
        self.encoder_outputs = None
        self.decoder_outputs = None
        self.projections = None
        self.fused = None
        self.hidden = None
        self.output = None

    @property
    def fusion_weights(self):
        w = getattr(self.fusion, "weights", None)
        if w is not None:
            w = nn.functional.softmax(w.squeeze(), dim=-1).detach()
        return w

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.projections = [self.projector(x) for x in self.encoder_outputs]
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)

        self.decoder_outputs = self.decoders(
            [inp.view(-1, *size) for inp, size in zip(self.encoder_outputs, self.encoders.output_sizes_before_flatten)]
        )
        return self.output


@register_model
class CAEKM(BaseModelKMeans, BaseModelPreTrain):
    def __init__(self, cfg):
        super(CAEKM, self).__init__(cfg, flatten_encoder_output=False)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes)

        self.latent_normalizer = get_normalizer(getattr(cfg, "latent_norm", None))

        if cfg.projector_config is not None:
            self.projector = Encoder(cfg.projector_config)
        else:
            self.projector = nn.Identity()

        self.views = None
        self.encoder_outputs = None
        self.latents = None
        self.decoder_outputs = None
        self.projections = None

    @property
    def eval_tensors(self):
        return th.cat(self.latents, dim=1)

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.latents = [self.latent_normalizer(th.flatten(x, start_dim=1)) for x in self.encoder_outputs]
        self.projections = [self.projector(lat) for lat in self.latents]
        self.decoder_outputs = self.decoders(self.encoder_outputs)

        return self.dummy_output(views[0].size(0))


class SelfExpressiveLayer(nn.Module):
    def __init__(self, n_samples):
        super(SelfExpressiveLayer, self).__init__()

        self.n_samples = n_samples
        initial_weights = th.full((n_samples, n_samples), 1e-4) - (1e-4 * th.eye(n_samples))
        self.register_parameter(
            "weight", nn.Parameter(data=initial_weights.to(device=config.DEVICE), requires_grad=True)
        )

    def weight_zero_diag(self):
        return self.weight - th.eye(self.n_samples).type_as(self.weight) * self.weight

    def forward(self, inp):
        w = self.weight_zero_diag()
        return w @ inp


@register_model
class DMSC(BaseModelPreTrain, BaseModelSpectral):
    def __init__(self, cfg):
        super(DMSC, self).__init__(cfg, flatten_encoder_output=False)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes)
        self.self_expressive = SelfExpressiveLayer(n_samples=cfg.n_samples)
        
        self.calc_self_representations = True
        self.views = None
        self.encoder_outputs = None
        self.latents = None
        self.decoder_outputs = None
        self.self_representations = None

    @property
    def affinity(self):
        abs_w = th.abs(self.self_expressive.weight_zero_diag())
        return (abs_w + abs_w.T) / 2

    def init_pre_train(self):
        super(DMSC, self).init_pre_train()
        self.calc_self_representations = False

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.decoder_outputs = self.decoders(self.encoder_outputs)

        if self.calc_self_representations:
            self.latents = [th.flatten(out, start_dim=1) for out in self.encoder_outputs]
            self.self_representations = [self.self_expressive(lat) for lat in self.latents]
        else:
            self.latents = self.self_representations = None

        return self.dummy_output(views[0].size(0))


class Discriminator(nn.Module):
    def __init__(self, cfg, input_size):
        super(Discriminator, self).__init__()
        self.mlp = Encoder(cfg.mlp_config, input_size=input_size)
        self.output_layer = nn.Sequential(
            nn.Linear(self.mlp.output_size[0], 1, bias=True),
            nn.Sigmoid()
        )
        self.d0 = self.dv = None

    def forward(self, x0, xv):
        self.d0 = self.output_layer(self.mlp(x0))
        self.dv = self.output_layer(self.mlp(xv))
        return [self.d0, self.dv]


class AttentionLayer(nn.Module):
    def __init__(self, cfg, input_size):
        super(AttentionLayer, self).__init__()
        self.tau = cfg.tau
        self.mlp = Encoder(cfg.mlp_config, input_size=[input_size[0] * cfg.n_views])
        self.output_layer = nn.Linear(self.mlp.output_size[0], cfg.n_views, bias=True)
        self.weights = None

    def forward(self, xs):
        h = th.cat(xs, dim=1)
        act = self.output_layer(self.mlp(h))
        e = nn.functional.softmax(th.sigmoid(act) / self.tau, dim=1)
        self.weights = th.mean(e, dim=0)
        return self.weights


class DummyAttentionLayer(nn.Module):
    @staticmethod
    def forward(xs):
        n_views = len(xs)
        weights = th.ones(n_views).type_as(xs[0]) / n_views
        return weights


@register_model
class EAMC(BaseModelPreTrain):
    def __init__(self, cfg):
        super(EAMC, self).__init__(cfg)

        assert all([self.encoders.output_sizes[0] == s for s in self.encoders.output_sizes])
        hidden_size = self.encoders.output_sizes[0]

        if cfg.attention_config is not None:
            self.attention = AttentionLayer(cfg.attention_config, input_size=hidden_size)
            self.fusion = None
            fused_size = hidden_size
        else:
            self.attention = None
            assert cfg.fusion_config is not None, "EAMC expects either attention_config or fusion_config to be not None"
            self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
            fused_size = self.fusion.output_size

        self.discriminators = nn.ModuleList(
            [Discriminator(cfg.discriminator_config, input_size=hidden_size)
             for _ in range(len(cfg.encoder_configs) - 1)]
            )

        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=fused_size)

        # Kernel widths for view-specific and fused representations
        self.encoder_kernel_width = kernel_width.get_kernel_width_module(
            cfg.encoder_kernel_width_config, input_size=None
        )
        self.fused_kernel_width = kernel_width.get_kernel_width_module(
            cfg.fused_kernel_width_config, input_size=None
        )

        self.encoder_outputs = None
        self.discriminator_outputs = None
        self.weights = None
        self.fused = None
        self.hidden = None
        self.output = None

    def configure_optimizers(self):
        # Optimizer for encoders, attention and clustering module
        groups = [
            {'params': self.encoders.parameters(), 'lr': 1e-5},
            {'params': self.clustering_module.parameters(), 'lr': 1e-5}
        ]
        if self.attention is not None:
            groups.append({'params': self.attention.parameters(), 'lr': 1e-4})
        else:
            groups.append({'params': self.fusion.parameters(), 'lr': 1e-4})

        enc = th.optim.Adam(groups, betas=(0.95, 0.999))
        # Optimizer for discriminator
        disc = th.optim.Adam(self.discriminators.parameters(), 1e-3, betas=(0.5, 0.999))
        return enc, disc

    def training_step(self, batch, batch_idx, optimizer_idx):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)
        losses = self.get_loss()

        if optimizer_idx == 0:
            # Train encoders, attention and clustering module.
            del losses["EAMCDiscriminator"]
            del losses["tot"]
        elif optimizer_idx == 1:
            # Train discriminator
            losses = {"EAMCDiscriminator": losses["EAMCDiscriminator"]}
        else:
            raise RuntimeError()

        losses["tot"] = sum(losses.values())
        self._log_dict(losses, prefix="train_loss")
        return losses["tot"]

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.discriminator_outputs = [
            self.discriminators[i](self.encoder_outputs[0], self.encoder_outputs[i + 1])
            for i in range(len(self.encoder_outputs) - 1)
        ]

        if self.attention is not None:
            self.weights = self.attention(self.encoder_outputs)
            self.fused = th.sum(self.weights[None, None, :] * th.stack(self.encoder_outputs, dim=-1), dim=-1)
        else:
            self.fused = self.fusion(self.encoder_outputs)

        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output


@register_model
class MIMVC(BaseModelPreTrain):
    def __init__(self, cfg):
        super(MIMVC, self).__init__(cfg)

        self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=self.fusion.output_size)

        self.encoder_outputs = None
        self.fused = None
        self.hidden = None
        self.output = None

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.encoder_outputs = [nn.functional.softmax(x, dim=1) for x in self.encoder_outputs]

        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output
