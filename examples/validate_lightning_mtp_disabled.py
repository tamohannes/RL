#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove the Lightning trainer override omits auxiliary MTP state."""

from __future__ import annotations

import argparse
import json

import torch
from nemo_automodel.components.models.nemotron_v3.model import (
    NemotronHForCausalLM,
)
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    source_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    assert source_layers == 1, (
        f"expected released Lightning config to declare one MTP layer, got {source_layers}"
    )

    with torch.device("meta"):
        model = NemotronHForCausalLM(
            config,
            num_nextn_predict_layers=0,
        )
    runtime_layers = int(getattr(model.mtp_config, "num_layers", -1))
    mtp_state_keys = [
        key for key in model.state_dict() if key.startswith("mtp.") or ".mtp." in key
    ]
    assert runtime_layers == 0
    assert model.mtp is None
    assert not mtp_state_keys
    assert int(getattr(config, "num_nextn_predict_layers", -1)) == source_layers
    print(
        json.dumps(
            {
                "status": "ok",
                "source_num_nextn_predict_layers": source_layers,
                "runtime_num_nextn_predict_layers": runtime_layers,
                "mtp_is_none": True,
                "mtp_state_keys": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
