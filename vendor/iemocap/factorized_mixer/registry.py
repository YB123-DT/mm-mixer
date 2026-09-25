MIXER_VARIANTS = {f"M{i}": {} for i in range(1, 5)}
MIXER_VARIANTS.update({"M1_LR05": {}, "M1_LR2": {}, "M4_LR05": {}, "M4_LR2": {}})
MIXER_VARIANTS.update({f"{base}_LR{lr}_D{drop}_H{hidden}": {}
                       for base in ("M1", "M4") for lr in ("05", "10", "2")
                       for drop in ("0", "2") for hidden in ("1024", "1536")})
MIXER_VARIANTS.update({"M2_LR2": {}, "M3_LR2": {}})
MIXER_VARIANTS.update({"H1": {}, "H2": {}})
MIXER_VARIANTS.update({f"{base}_K{k}_D{d}_L{l}_LR{lr}_P{p}": {}
                       for base in ("M4",) for k in (4,6,8,12,16) for d in (128,256,384)
                       for l in (2,4) for lr in (1,2) for p in (0,2)})
MIXER_VARIANTS.update({f"ABL_{name}": {} for name in ("S","C","F","SC","SF","CF","SCF")})
MIXER_VARIANTS.update({f"LS_{name}": {} for name in ("01","001","learnable")})
MIXER_VARIANTS.update({"M4_BD05": {}, "M4_BD10": {}})
MIXER_VARIANTS.update({"M4_NOAUX_LR2_D2_H1536": {}})
MIXER_VARIANTS.update({"HYB_M4_LR2_D2_H1536": {}})
MIXER_VARIANTS.update({f"HO_WO_TAV_KR{ratio}": {} for ratio in (1, 4, 8, 16)})
MIXER_VARIANTS.update({f"HO_WO_TAV_MR{ratio}": {} for ratio in (2, 4, 8)})
MIXER_VARIANTS.update({"M4_CA_PARALLEL_X3": {}})
MIXER_VARIANTS.update({name: {} for name in (
    "HO_AV_ONLY", "HO_TV_ONLY", "HO_TA_ONLY", "HO_TAV_ONLY",
    "HO_WO_AV", "HO_WO_TV", "HO_WO_TA", "HO_WO_TAV", "HO_WO_TAV_CSSAV",
)})
MIXER_VARIANTS.update({"M4_WOTAV_NO_ADAPTIVE_CHANNEL": {}})
MIXER_VARIANTS.update({"HO_RAW_AUX_FULL": {}})
MIXER_VARIANTS.update({name: {} for name in (
    "M4_BS64_LR1", "M4_BS64_LRSQRT2", "M4_BS64_LR2", "M4_BS64_LR4",
)})
MIXER_VARIANTS.update({name: {} for name in (
    "BLOCK_TRANSFORMER_CTRL", "BLOCK_RESIDUAL_ONLY", "BLOCK_MIXER_ONLY",
)})
MIXER_VARIANTS.update({f"RS_R{i}": {} for i in range(1, 8)})
MIXER_VARIANTS.update({"FG_IDENTITY": {}, "FG_TRANSFORMER": {}})
MIXER_VARIANTS.update({f"FG_CAP_H{hidden}": {} for hidden in (128, 256, 384, 512)})
MIXER_VARIANTS.update({"AUX_PRE_FEATURE_GATE": {}})
MIXER_VARIANTS.update({f"CORE_C{width}": {} for width in (320, 384, 512)})
MIXER_VARIANTS.update({name: {} for name in (
    "FINAL_F0_FULL", "FINAL_F1_NO_SUBSPACE", "FINAL_F2_NO_ROUTING",
    "FINAL_F3_NO_FFN", "FINAL_F4_NO_ALTERNATING", "FINAL_F5_NO_MIXER",
    "FINAL_F6_NO_X3", "FINAL_F7_TRANSFORMER", "FINAL_F8_TR_X3_MIXER1",
    "FINAL_F9_X3_TRANSFORMER_REPLAY", "FINAL_F10_NO_CA",
)})
MIXER_VARIANTS.update({"FINAL_INFO_GATE_ON": {}})

LIGHT_CAPACITY_VARIANTS = {
    "LC_C1_H1792", "LC_C2_H2048", "LC_C3_L3",
    "LC_C4_POST_TR", "LC_C5_POST_MLP",
}
MIXER_VARIANTS.update({name: {} for name in LIGHT_CAPACITY_VARIANTS})

C4_TUNE30_ARCH_VARIANTS = {
    "LC30_TR_H4_F512_D10", "LC30_TR_H16_F512_D10",
    "LC30_TR_H8_F256_D10", "LC30_TR_H8_F768_D10", "LC30_TR_H8_F1024_D10",
    "LC30_TR_H8_F512_D00", "LC30_TR_H8_F512_D05",
    "LC30_TR_H8_F512_D15", "LC30_TR_H8_F512_D20",
    "LC30_TR_H8_F768_D05",
}
MIXER_VARIANTS.update({name: {} for name in C4_TUNE30_ARCH_VARIANTS})


FINAL_TUNE_VARIANTS = {
    f"TUNE_R1_R{rank}_S{int(scale * 100):02d}_LR{int(lr * 100):02d}": {
        "x3_rank": rank,
        "x3_residual_scale": scale,
        "x3_lr_multiplier": lr,
    }
    for rank in (16, 24, 32, 48)
    for scale in (.75, 1.0, 1.25)
    for lr in (.5, .75, 1.0, 1.5)
}
FINAL_TUNE_VARIANTS.update({
    f"TUNE_R2_H{heads}_D{int(dropout * 100):02d}_Q{int(query * 100):02d}": {
        "ca_heads": heads,
        "ca_dropout": dropout,
        "ca_query_bypass": query,
    }
    for heads in (4, 8, 16)
    for dropout in (0.0, .1, .2, .3)
    for query in (0.0, .1, .2, .3)
})
FINAL_TUNE_VARIANTS.update({
    f"TUNE_R3_AD{int(adaptive * 100):02d}_CD{int(channel_dropout * 100):02d}_CR{int(channel_residual * 100):02d}": {
        "adaptive_dropout": adaptive,
        "channel_dropout": channel_dropout,
        "channel_residual": channel_residual,
    }
    for adaptive in (0.0, .05, .1, .15)
    for channel_dropout in (0.0, .1, .2, .3)
    for channel_residual in (0.0, .05, .1)
})
FINAL_TUNE_VARIANTS.update({
    f"TUNE_R4_MS_{candidate_id}": {
        "mixer_sub_scale": subspace,
        "mixer_mod_scale": modality,
        "mixer_ffn_scale": ffn,
    }
    for candidate_id, (subspace, modality, ffn) in {
        "M00": (1.0, 1.0, 1.0),
        "M01": (0.5, 1.0, 1.0),
        "M02": (0.75, 1.0, 1.0),
        "M03": (1.25, 1.0, 1.0),
        "M04": (1.0, 0.5, 1.0),
        "M05": (1.0, 0.75, 1.0),
        "M06": (1.0, 1.25, 1.0),
        "M07": (1.0, 1.0, 0.5),
        "M08": (1.0, 1.0, 0.75),
        "M09": (1.0, 1.0, 1.25),
        "M10": (0.75, 0.75, 0.75),
        "M11": (1.25, 1.25, 1.25),
    }.items()
})
MIXER_VARIANTS.update({name: {} for name in FINAL_TUNE_VARIANTS})
