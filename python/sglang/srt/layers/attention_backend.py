from abc import ABC

import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

from sglang.global_config import global_config


class AttentionBackend(ABC):
    """The base class of attention backends"""

    pass


class FlashInferAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        super().__init__()

        if not _grouped_size_compiled_for_decode_kernels(
            model_runner.model_config.num_attention_heads // model_runner.tp_size,
            model_runner.model_config.get_num_kv_heads(model_runner.tp_size),
        ):
            use_tensor_cores = True
        else:
            use_tensor_cores = False

        self.flashinfer_workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device="cuda",
        )
        if model_runner.sliding_window_size is None:
            self.flashinfer_prefill_wrapper_ragged = (
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.flashinfer_workspace_buffer, "NHD"
                )
            )
            self.flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                self.flashinfer_workspace_buffer, "NHD"
            )
            self.flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.flashinfer_workspace_buffer,
                "NHD",
                use_tensor_cores=use_tensor_cores,
            )
        else:
            # Two wrappers: one for full attention and one for sliding window attention.
            self.flashinfer_prefill_wrapper_paged = []
            self.flashinfer_decode_wrapper = []
            for _ in range(2):
                self.flashinfer_prefill_wrapper_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer, "NHD"
                    )
                )
                self.flashinfer_decode_wrapper.append(
                    BatchDecodeWithPagedKVCacheWrapper(
                        self.flashinfer_workspace_buffer,
                        "NHD",
                        use_tensor_cores=use_tensor_cores,
                    )
                )


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner):
        super().__init__()
