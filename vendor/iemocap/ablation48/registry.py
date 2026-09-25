from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .lock import validate_results_lock


SOURCE_SHA = "11638d32f3eda53602a9f8d184b47e1f9c8188601d83702b67b3811af2f08641"
DATA_SHA = "bee09ed676bb2bbd6859a93019e70e428467af95d1d82251d516de83c3005c24"
FINAL_ARTIFACTS = Path(__file__).resolve().parents[1] / "final_artifacts"
RESULTS_LOCK_PATH = FINAL_ARTIFACTS / "results_lock.json"
SELECTION_48_PATH = FINAL_ARTIFACTS / "48_selection.json"
VENDORED_BASE_SOURCE = str(
    Path(__file__).resolve().parents[1] / "base/multiattn.py"
)


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 3e-5
    batch_size: int = 32
    epochs: int = 100
    seed: int = 2025


@dataclass(frozen=True)
class BatchSchedule:
    temporal_encode_scope: str = "full_dialogue_first"
    turn_chunk_size: int = 32
    train_turns: int = 5810
    chunks_per_epoch: int = 182
    gradient_accumulation: int = 2
    optimizer_updates_per_epoch: int = 91
    ema_updates_per_epoch: int = 91
    ema_update_unit: str = "optimizer_step"


@dataclass(frozen=True)
class ModuleKwarg:
    name: str
    value: bool | int | float | str


@dataclass(frozen=True)
class CanonicalConfig:
    k: int = 4
    token_dim: int = 128
    layers: int = 1
    ffn_dim: int = 512
    heads: int = 8
    tokenizer: str = "linear_split"
    mixer: str = "self_attention"
    topology: str = "full"
    aggregator: str = "mean_project"
    information_gate: str = "pre_cross_original"
    channel_gate: str = "original"
    readout: str = "query3_dense"
    classifier: str = "linear3"
    regularizer: str = "none"
    dropout: str = "original"
    trainable_scope: str = "rawaux_all"
    batch_protocol: str = "utterance"
    temporal_protocol: str = "none"
    reads_dialogue_metadata: bool = False
    optimizer: OptimizerConfig = OptimizerConfig()
    batch_schedule: BatchSchedule = BatchSchedule()
    module_kwargs: tuple[ModuleKwarg, ...] = ()


S0_CONFIG = CanonicalConfig()


@dataclass(frozen=True)
class ExperimentRow:
    experiment_id: str
    title: str
    family: str
    config: CanonicalConfig
    declared_delta_fields: tuple[str, ...]
    matched_control: str
    compatibility_owner: str
    owned_boundaries: frozenset[str]
    tensor_contracts: tuple[str, ...]
    trainable_scope: str
    parameter_count: int | None
    parameter_count_status: str
    source_path: str
    source_sha256: str
    data_path: str
    data_sha256: str


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        out = {}
        for f in fields(value):
            key = f"{prefix}.{f.name}" if prefix else f.name
            out.update(_flatten(getattr(value, f.name), key))
        return out
    return {prefix: value}


def canonical_diff(a: CanonicalConfig, b: CanonicalConfig) -> frozenset[str]:
    left, right = _flatten(a), _flatten(b)
    return frozenset(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))


def _cfg(**changes: Any) -> CanonicalConfig:
    return replace(S0_CONFIG, **changes)


def _kwargs(**values: bool | int | float | str) -> tuple[ModuleKwarg, ...]:
    return tuple(ModuleKwarg(name, value) for name, value in sorted(values.items()))


_DEFINITIONS: dict[int, tuple[str, str, str, dict[str, Any], str]] = {
    1:("independent slot MLPs","tokenizer","tokenizer",{"tokenizer":"independent_mlp"},"S0"),
    2:("shared projector and private adapters","tokenizer","tokenizer",{"tokenizer":"shared_projector_private_adapter"},"S0"),
    3:("modality-conditioned latent queries","tokenizer","tokenizer",{"tokenizer":"latent_cross_attention"},"S0"),
    4:("competitive slot normalization","tokenizer","tokenizer",{"tokenizer":"competitive_linear"},"S0"),
    5:("soft token importance","routing","routing",{"mixer":"soft_importance"},"S0"),
    6:("ST Gumbel top-2","routing","routing",{"mixer":"gumbel_top2","module_kwargs":_kwargs(temperature_start=1.0,temperature_end=.5,anneal_epochs=30,top_k=2)},"S0"),
    7:("TokenFusion replacement","routing","routing",{"mixer":"tokenfusion"},"S0"),
    8:("fusion bottlenecks","routing","routing",{"mixer":"bottleneck4"},"S0"),
    9:("post-cross InformationGate","gate","gate",{"information_gate":"post_cross_original"},"S0"),
    10:("post-aggregation InformationGate","gate","gate",{"information_gate":"post_aggregation_original"},"S0"),
    11:("post-cross FiLM","gate","gate",{"information_gate":"post_cross_film"},"S0"),
    12:("post-cross bounded MAG","gate","gate",{"information_gate":"post_cross_mag_0.1","module_kwargs":_kwargs(max_residual_scale=.1)},"S0"),
    13:("L2 ReZero","depth","depth",{"layers":2,"mixer":"rezero"},"S1P"),
    14:("L2 LayerScale","depth","depth",{"layers":2,"mixer":"layerscale_1e-5"},"S1P"),
    15:("L2 Re-attention","depth","depth",{"layers":2,"mixer":"reattention"},"S1P"),
    16:("L2 PreLN SwiGLU","depth","depth",{"layers":2,"mixer":"preln_swiglu"},"S1P"),
    17:("cross-modal-only topology","topology","routing",{"topology":"cross_modal_only"},"S0"),
    18:("same-slot cross-modal topology","topology","routing",{"topology":"same_slot_cross_modal"},"S0"),
    19:("two-stage topology","topology","routing",{"topology":"cross_then_within"},"S0"),
    20:("relation tokens","topology","routing",{"tokenizer":"pair_relation","topology":"relation_only"},"S0"),
    21:("learned weighted aggregation","aggregator","aggregator",{"aggregator":"weighted_sum","module_kwargs":_kwargs(weight_sum_scale=1.0)},"S0"),
    22:("PMA aggregation","aggregator","aggregator",{"aggregator":"pma"},"S0"),
    23:("residual concat aggregation","aggregator","aggregator",{"aggregator":"residual_concat_alpha0"},"S0"),
    24:("residual PMA aggregation","aggregator","aggregator",{"aggregator":"residual_pma_alpha0"},"S0"),
    25:("residual SwiGLU integrator","readout","readout",{"readout":"query3_residual_swiglu","module_kwargs":_kwargs(hidden_dim=1024)},"S0"),
    26:("global PMA residual MLP","readout","readout",{"readout":"global_pma_residual_mlp","module_kwargs":_kwargs(hidden_dim=512)},"S0"),
    27:("hierarchical attention pooling","readout","readout",{"readout":"hierarchical_12_3_1"},"S0"),
    28:("six emotion queries","readout","readout",{"readout":"emotion_queries6","classifier":"class_query_scalar","module_kwargs":_kwargs(emotion_label_order="happiness,sadness,neutral,anger,excited,frustration")},"S0"),
    29:("one shared three private","shared_private","tokenizer",{"tokenizer":"shared1_private3"},"S0"),
    30:("two shared two private","shared_private","tokenizer",{"tokenizer":"shared2_private2"},"S0"),
    31:("three shared one private","shared_private","tokenizer",{"tokenizer":"shared3_private1"},"S0"),
    32:("shared/private decorrelation","shared_private","tokenizer",{"tokenizer":"shared2_private2","regularizer":"decorrelation_0.01","module_kwargs":_kwargs(lambda_decorrelation=.01)},"S0"),
    33:("heterogeneous centered views","classifier","readout",{"classifier":"global_text_av_heads"},"S0"),
    34:("pair relation heads","classifier","readout",{"classifier":"ta_tv_av_relation_heads","module_kwargs":_kwargs(relation_formula="concat(hi,hj,hi*hj,abs(hi-hj)):1024->512->256")},"S0"),
    35:("sample-conditioned view router","classifier","readout",{"classifier":"routed_global_text_av_heads"},"S0"),
    36:("residual nonlinear classifier","classifier","readout",{"classifier":"residual_nonlinear_single"},"S0"),
    37:("modality dropout .1","regularization","dropout",{"dropout":"modality_0.1"},"S0"),
    38:("modality dropout .2","regularization","dropout",{"dropout":"modality_0.2"},"S0"),
    39:("token dropout .1","regularization","dropout",{"dropout":"token_0.1"},"S0"),
    40:("entropy quality bias","regularization","quality",{"regularizer":"entropy_quality_bias_0.1","module_kwargs":_kwargs(lambda_quality=.1,quality_stop_gradient=True)},"S0"),
    41:("audio causal context","temporal","temporal",{"batch_protocol":"dialogue","temporal_protocol":"audio_causal","reads_dialogue_metadata":True,"module_kwargs":_kwargs(residual_alpha_init=0.0,temporal_layers=1,turn_chunk_size=32,gradient_accumulation=2,optimizer_updates_per_epoch=91,ema_updates_per_epoch=91)},"DIALOGUE_NULL"),
    42:("visual causal context","temporal","temporal",{"batch_protocol":"dialogue","temporal_protocol":"visual_causal","reads_dialogue_metadata":True,"module_kwargs":_kwargs(residual_alpha_init=0.0,temporal_layers=1,turn_chunk_size=32,gradient_accumulation=2,optimizer_updates_per_epoch=91,ema_updates_per_epoch=91)},"DIALOGUE_NULL"),
    43:("audio visual causal context","temporal","temporal",{"batch_protocol":"dialogue","temporal_protocol":"av_causal","reads_dialogue_metadata":True,"module_kwargs":_kwargs(residual_alpha_init=0.0,temporal_layers=1,turn_chunk_size=32,gradient_accumulation=2,optimizer_updates_per_epoch=91,ema_updates_per_epoch=91)},"DIALOGUE_NULL"),
    44:("audio visual full context","temporal","temporal",{"batch_protocol":"dialogue","temporal_protocol":"av_bidirectional","reads_dialogue_metadata":True,"module_kwargs":_kwargs(residual_alpha_init=0.0,temporal_layers=1,turn_chunk_size=32,gradient_accumulation=2,optimizer_updates_per_epoch=91,ema_updates_per_epoch=91)},"DIALOGUE_NULL"),
    45:("parameter-matched three-token transformer","control","control",{"k":1,"token_dim":256,"ffn_dim":192,"tokenizer":"identity_three_token","aggregator":"identity","module_kwargs":_kwargs(parameter_tolerance=.01,target_subspace_params=362880)},"S0"),
    46:("expanded tokens without mixer","control","control",{"mixer":"identity"},"S0"),
}


def build_registry() -> Mapping[str, ExperimentRow]:
    rows = {}
    contract = ("h_cross[B,3,256]", "z0[B,3,K,d]", "z_enc[B,3K,d]", "logits[B,6]")
    for number in range(1, 47):
        title, family, owner, changes, control = _DEFINITIONS[number]
        config = _cfg(**changes)
        delta = tuple(sorted(canonical_diff(S0_CONFIG, config)))
        key = f"{number:02d}"
        boundary_map = {
            "tokenizer": frozenset({"tokenizer"}), "routing": frozenset({"routing"}),
            "gate": frozenset({"gate"}), "depth": frozenset({"depth", "routing"}),
            "aggregator": frozenset({"aggregator"}), "readout": frozenset({"readout"}),
            "dropout": frozenset({"dropout"}), "quality": frozenset({"quality"}),
            "temporal": frozenset({"temporal"}), "control": frozenset({"control"}),
        }
        owned = boundary_map[owner]
        if number == 20:
            owned = frozenset({"tokenizer", "routing"})
        elif number == 45:
            owned = frozenset({"tokenizer", "aggregator", "depth"})
        elif number == 46:
            owned = frozenset({"routing"})
        rows[key] = ExperimentRow(
            key, title, family, config, delta, control, owner, owned, contract,
            config.trainable_scope,
            None,
            "pending_formal_measurement",
            VENDORED_BASE_SOURCE, SOURCE_SHA,
            "/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz",
            DATA_SHA,
        )
    return MappingProxyType(rows)


def compatible(a: ExperimentRow, b: ExperimentRow) -> bool:
    if a.config.batch_protocol != "utterance" or b.config.batch_protocol != "utterance":
        return False
    exclusive = {"tokenizer", "routing", "gate", "depth", "aggregator", "readout", "dropout"}
    return (a.owned_boundaries & b.owned_boundaries & exclusive) == frozenset()


def compatibility_matrix(registry: Mapping[str, ExperimentRow] | None = None):
    registry = build_registry() if registry is None else registry
    return MappingProxyType({(a, b): compatible(registry[a], registry[b])
                             for a in registry for b in registry})


def materialize_combination(experiment_id: str, results_lock: Mapping[str, Any] | None):
    if experiment_id not in {"47", "48"}:
        raise ValueError("only deferred experiments 47 and 48 can be materialized")
    if results_lock is None:
        raise RuntimeError("results_lock.json with SHA-256 is required before materializing 47/48")
    validate_results_lock(results_lock)
    canonical=json.loads(RESULTS_LOCK_PATH.read_text())
    validate_results_lock(canonical)
    if results_lock != canonical:
        raise ValueError("provided lock is not the canonical locked results")
    selection=json.loads(SELECTION_48_PATH.read_text())
    file_sha=hashlib.sha256(RESULTS_LOCK_PATH.read_bytes()).hexdigest()
    if selection.get("results_lock_sha256") != file_sha:
        raise ValueError("48 selection is not bound to the canonical results-lock file SHA")
    if experiment_id == "47":
        raise RuntimeError("47 is NO-GO_NO_POSITIVE_CORE_PAIR")
    if (selection.get("experiment_id"),selection.get("status"),selection.get("selection_frozen"),selection.get("members")) != (
            "48","MATERIALIZED",True,["16","40"]):
        raise ValueError("48 frozen selection contract is invalid")
    config=_cfg(layers=2,mixer="preln_swiglu",regularizer="entropy_quality_bias_0.1",
                module_kwargs=_kwargs(lambda_quality=.1,quality_stop_gradient=True,locked_members="16,40"))
    return ExperimentRow(
        "48","locked combination 16+40","deferred_combination",config,
        tuple(sorted(canonical_diff(S0_CONFIG,config))),"S1P","combination",
        frozenset({"depth","routing","quality"}),
        ("h_cross[B,3,256]","z0[B,3,4,128]","z_enc[B,12,128]","logits[B,6]"),
        config.trainable_scope,None,"pending_formal_measurement",
        VENDORED_BASE_SOURCE,SOURCE_SHA,
        "/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz",DATA_SHA,
    )


def materialize_locked_48() -> ExperimentRow:
    return materialize_combination("48",json.loads(RESULTS_LOCK_PATH.read_text()))
