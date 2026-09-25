"""Explicit construction; no process-global active experiment state."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from torch import nn

from .e14 import SubspaceTokenEncoder, ThreeTokenWideEncoder
from .modules_01_12 import (
    BottleneckMixer, CompetitiveTokenizer, FamilyEncoder, GumbelTop2Mixer,
    IdentityInfo, IdentityMixer, IndependentMLPTokenizer, LatentQueryTokenizer, LinearTokenizer,
    OriginalGateBlock, PostCrossFiLM, PostCrossMAG, SharedProjectorTokenizer,
    SoftImportanceMixer, TokenFusionMixer,
)
from .modules_13_24 import Family1324Encoder
from .modules_25_36 import (
    ClassQueryScalars, EmotionQueryPool, EmotionTokenIdentity, GlobalPMA,
    HierarchicalPoolEncoder, ObservableSubspaceEncoder, PassThroughPool,
    RawEmotionTokenEncoder, RelationHead, ResidualMLP256,
    ResidualNonlinearClassifier, ResidualSwiGLUIntegrator,
    RoutedViewClassifier, SharedPrivateEncoder, ViewHead,
)
from .modules_37_46 import (
    DialogueTemporalWrapper, IdentityMixerSubspaceEncoder, TokenDropSubspaceEncoder,
    install_modality_dropout, install_quality_bias,
)
from .registry import materialize_locked_48

BASE_PATH = Path(__file__).resolve().parents[1] / "base/multiattn.py"


def _load_base():
    spec = importlib.util.spec_from_file_location("rawaux48_pinned_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_fusion_model(experiment_id: str, dropout: float = .1):
    if experiment_id not in {"S0", "48", *(f"{i:02d}" for i in range(1, 47))}:
        raise NotImplementedError(f"experiment {experiment_id} is not implemented")
    base = _load_base()
    model = base.HierarchicalAttentionFusion(
        embed_dims={"v": 342, "a": 1024, "t": 1024}, num_classes=6,
        modalities=["v", "a", "t"], fusion_dim=256, num_transformer_layers=3,
        num_heads=8, dropout=dropout,
        modality_importance={"t": .65, "a": .2, "v": .15},
        use_moe=False, use_contrastive=False, aux_from_raw_projected=True,
        align_self_weight=.95,
    )
    model.disable_alignment()
    if experiment_id == "48":
        row=materialize_locked_48()
        members={item.name:item.value for item in row.config.module_kwargs}["locked_members"]
        if members != "16,40": raise RuntimeError("locked 48 members changed")
        model.transformer_encoder=Family1324Encoder("16",dropout,tuple(model.modalities))
        install_quality_bias(model,.1,copy_safe=True)
        model.capacity_variant="48"; model.combination_members=("16","40")
        model.combination_install_order=("16:Family1324Encoder","40:quality_bias_0.1")
        return model
    if experiment_id == "S0":
        model.transformer_encoder = SubspaceTokenEncoder(dropout)
        model.capacity_variant = "E14"
        return model

    if 37 <= int(experiment_id) <= 46:
        eid=int(experiment_id)
        if eid in {37,38}:
            model.transformer_encoder=SubspaceTokenEncoder(dropout)
            install_modality_dropout(model,.1 if eid==37 else .2)
        elif eid==39:
            model.transformer_encoder=TokenDropSubspaceEncoder(dropout,.1)
        elif eid==40:
            model.transformer_encoder=SubspaceTokenEncoder(dropout); install_quality_bias(model,.1)
        elif eid in {41,42,43,44}:
            model.transformer_encoder=SubspaceTokenEncoder(dropout)
            modalities={41:("a",),42:("v",),43:("a","v"),44:("a","v")}[eid]
            model=DialogueTemporalWrapper(model,modalities,causal=eid!=44,microbatch=32)
        elif eid==45:
            model.transformer_encoder=ThreeTokenWideEncoder(dropout)
            model.transformer_encoder.token_count=1
        else:
            model.transformer_encoder=IdentityMixerSubspaceEncoder(dropout)
        model.capacity_variant=experiment_id
        return model

    if 25 <= int(experiment_id) <= 36:
        eid=int(experiment_id)
        if eid==25:
            model.transformer_encoder=SubspaceTokenEncoder(dropout)
            model.feature_integrator=ResidualSwiGLUIntegrator(1024,dropout)
        elif eid==26:
            model.transformer_encoder=SubspaceTokenEncoder(dropout)
            model.pool=GlobalPMA(dropout); model.pool_queries.requires_grad_(False)
            model.feature_integrator=ResidualMLP256(dropout)
        elif eid==27:
            model.transformer_encoder=HierarchicalPoolEncoder(dropout)
            model.pool=PassThroughPool(); model.pool_queries.requires_grad_(False)
            model.feature_integrator=nn.Identity()
        elif eid==28:
            model.transformer_encoder=RawEmotionTokenEncoder(dropout)
            model.pool=EmotionQueryPool(dropout); model.pool_queries.requires_grad_(False)
            model.feature_integrator=EmotionTokenIdentity()
            model.classifiers=nn.ModuleList([ClassQueryScalars()]); model.num_classifier_heads=1
        elif eid in {29,30,31,32}:
            counts={29:(1,3),30:(2,2),31:(3,1),32:(2,2)}[eid]
            model.transformer_encoder=SharedPrivateEncoder(*counts,dropout,use_decorrelation=eid==32)
        elif eid in {33,34,35}:
            encoder=ObservableSubspaceEncoder(dropout); model.transformer_encoder=encoder
            if eid==33: model.classifiers=nn.ModuleList([ViewHead(encoder,n) for n in ("global","text","mean_av")])
            elif eid==34: model.classifiers=nn.ModuleList([RelationHead(encoder,n) for n in ("TA","TV","AV")])
            else: model.classifiers=nn.ModuleList([RoutedViewClassifier(encoder)])
            model.num_classifier_heads=len(model.classifiers)
        else:
            model.transformer_encoder=SubspaceTokenEncoder(dropout)
            model.classifiers=nn.ModuleList([ResidualNonlinearClassifier()]); model.num_classifier_heads=1
        model.capacity_variant=experiment_id
        return model

    if 13 <= int(experiment_id) <= 24:
        # The relation tokenizer derives TA/TV/AV from the base model's actual
        # tensor order instead of assuming a conceptual T/A/V order.
        model.transformer_encoder = Family1324Encoder(experiment_id, dropout, tuple(model.modalities))
        model.capacity_variant = experiment_id
        return model

    tokenizer = LinearTokenizer()
    mixer: nn.Module = IdentityMixer()
    if experiment_id == "01": tokenizer = IndependentMLPTokenizer()
    elif experiment_id == "02": tokenizer = SharedProjectorTokenizer()
    elif experiment_id == "03": tokenizer = LatentQueryTokenizer()
    elif experiment_id == "04": tokenizer = CompetitiveTokenizer()
    elif experiment_id == "05": mixer = SoftImportanceMixer()
    elif experiment_id == "06": mixer = GumbelTop2Mixer()
    elif experiment_id == "07": mixer = TokenFusionMixer()
    elif experiment_id == "08": mixer = BottleneckMixer()

    regulation = None; regulation_stage = None
    if experiment_id in {"09", "10", "11", "12"}:
        # The original pre-cross gate is removed, never stacked with the new hook.
        original = model.info_gates
        model.info_gates = nn.ModuleDict({m: IdentityInfo() for m in model.modalities})
        regulation_stage = "post_aggregation" if experiment_id == "10" else "post_cross"
        if experiment_id in {"09", "10"}: regulation = OriginalGateBlock(original, regulation_stage)
        elif experiment_id == "11": regulation = PostCrossFiLM()
        else: regulation = PostCrossMAG(.1)
        model.information_gate_stage = regulation_stage
        model.pre_information_gate_enabled = False

    # AdaptiveFusionGate already computes the required sample context. Capture it
    # on this model instance (not process-global state) for the immediately
    # downstream regulation hook.
    context_box = {"value": None}
    if regulation is not None:
        def save_context(_module, _inputs, output): context_box["value"] = output[0]
        model.adaptive_fusion.register_forward_hook(save_context)
    provider = (lambda: context_box["value"]) if regulation is not None else None
    model.transformer_encoder = FamilyEncoder(tokenizer, mixer, dropout, regulation, regulation_stage, provider)
    if regulation is not None:
        # Inspection alias only. Bypass nn.Module.__setattr__ so the regulator has
        # exactly one registered owner and therefore one checkpoint key prefix.
        object.__setattr__(model, "regulation_block", regulation)
    model.capacity_variant = experiment_id
    return model


def optimizer_parameter_groups(model, lr: float):
    partitions = (("proj.", lr * .5), ("transformer_encoder.", lr), ("classifiers.", lr * 2))
    groups, used = [], set()
    named = list(model.named_parameters())
    for prefix, group_lr in partitions:
        params = [p for name, p in named if name.startswith(prefix) and p.requires_grad]
        used.update(id(p) for p in params)
        if params:
            groups.append({"params": params, "lr": group_lr})
    remaining = [p for _, p in named if p.requires_grad and id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": lr})
    return groups


def experiment_losses(model):
    """Return structure-specific diagnostics separately from the RawAux loss."""
    encoder=model.transformer_encoder
    return encoder.experiment_losses() if hasattr(encoder,"experiment_losses") else {}


def experiment_training_loss(model):
    """The one scalar to add to the task objective; raw diagnostics are excluded."""
    losses=experiment_losses(model)
    if "decorrelation_weighted" in losses:
        return losses["decorrelation_weighted"]
    return next(model.parameters()).new_zeros(())
