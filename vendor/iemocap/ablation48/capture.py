from __future__ import annotations

from collections import OrderedDict

import torch


class CanonicalStageCapture:
    """Read-only hook capture of the canonical S0/E14 tensor flow."""
    ORDER = ("h_proj", "context", "h_info", "h_channel", "h_cross", "z0", "z",
             "z_enc", "h_sub", "pooled", "h_global", "logits")

    def __init__(self, model):
        self.model = model
        self._handles = []
        self._raw = {}

    @staticmethod
    def _tensor(value):
        if isinstance(value, tuple):
            value = value[0]
        return value.detach().clone()

    def _save(self, name):
        def hook(_module, _inputs, output):
            self._raw[name] = self._tensor(output)
        return hook

    def __enter__(self):
        m = self.model
        for modality in m.modalities:
            self._handles.append(m.feature_selectors[modality].register_forward_hook(
                self._save(f"proj_{modality}")))
            self._handles.append(m.info_gates[modality].register_forward_hook(
                self._save(f"info_{modality}")))
            self._handles.append(m.gates[modality].register_forward_hook(
                self._save(f"gate_{modality}")))
        self._handles.append(m.adaptive_fusion.register_forward_hook(self._save("context")))
        encoder = m.transformer_encoder
        self._handles.append(encoder.register_forward_pre_hook(
            lambda _module, inputs: self._raw.__setitem__("h_cross", self._tensor(inputs[0]))))
        self._handles.append(encoder.split.register_forward_hook(self._save("split")))
        self._handles.append(encoder.transformer.register_forward_pre_hook(
            lambda _module, inputs: self._raw.__setitem__("z", self._tensor(inputs[0]))))
        self._handles.append(encoder.transformer.register_forward_hook(self._save("z_enc")))
        self._handles.append(encoder.output_adapter.register_forward_hook(self._save("h_sub")))
        self._handles.append(m.pool.register_forward_hook(self._save("pooled")))
        self._handles.append(m.feature_integrator.register_forward_hook(self._save("h_global")))
        self._handles.append(m.register_forward_hook(
            lambda _module, _inputs, output: self._raw.__setitem__("logits", self._tensor(output[0]))))
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def stages(self):
        modalities = self.model.modalities
        h_proj = torch.stack([self._raw[f"proj_{m}"] for m in modalities], 1)
        h_info = torch.stack([self._raw[f"info_{m}"] for m in modalities], 1)
        gates = torch.stack([self._raw[f"gate_{m}"] for m in modalities], 1)
        split = self._raw["split"]
        batch = split.shape[0]
        values = {
            "h_proj": h_proj,
            "context": self._raw["context"],
            "h_info": h_info,
            "h_channel": h_info * gates + h_info * .1,
            "h_cross": self._raw["h_cross"],
            "z0": split.reshape(batch, 3, 4, 128),
            "z": self._raw["z"],
            "z_enc": self._raw["z_enc"],
            "h_sub": self._raw["h_sub"],
            "pooled": self._raw["pooled"],
            "h_global": self._raw["h_global"],
            "logits": self._raw["logits"],
        }
        return OrderedDict((name, values[name]) for name in self.ORDER)
