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

"""Small CUDA IPC check matching NeMo RL's raw-handle refit path."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys


def _set_alloc_conf(value: str) -> None:
    if value == "inherit":
        return
    if value == "unset":
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        return
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value


def _producer(handle_queue: mp.Queue, release: mp.Event, alloc_conf: str) -> None:
    _set_alloc_conf(alloc_conf)
    import torch

    from nemo_rl.models.policy.utils import get_handle_from_tensor

    torch.cuda.set_device(0)
    tensor = torch.arange(16, device="cuda", dtype=torch.float32)
    handle_queue.put(get_handle_from_tensor(tensor))
    if not release.wait(timeout=30):
        raise TimeoutError("consumer did not release the producer tensor")


def _consumer(
    handle_queue: mp.Queue,
    result_queue: mp.Queue,
    release: mp.Event,
    warmup: bool,
    alloc_conf: str,
) -> None:
    _set_alloc_conf(alloc_conf)
    import torch

    from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

    torch.cuda.set_device(0)
    if warmup:
        torch.empty(1, device="cuda")
    try:
        handle = handle_queue.get(timeout=30)
        tensor = rebuild_cuda_tensor_from_ipc(handle, 0)
        torch.cuda.synchronize()
        result_queue.put({"status": "pass", "values": tensor.cpu().tolist()})
    except Exception as error:
        result_queue.put(
            {
                "status": "error",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
    finally:
        release.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-consumer", action="store_true")
    parser.add_argument("--producer-alloc-conf", default="inherit")
    parser.add_argument("--consumer-alloc-conf", default="inherit")
    parser.add_argument(
        "--expect",
        choices=("pass", "pidfd-ebadf"),
        required=True,
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    handle_queue = mp.Queue()
    result_queue = mp.Queue()
    release = mp.Event()
    producer = mp.Process(
        target=_producer,
        args=(handle_queue, release, args.producer_alloc_conf),
    )
    consumer = mp.Process(
        target=_consumer,
        args=(
            handle_queue,
            result_queue,
            release,
            args.warmup_consumer,
            args.consumer_alloc_conf,
        ),
    )

    producer.start()
    consumer.start()
    try:
        result = result_queue.get(timeout=45)
    except queue.Empty:
        result = {"status": "error", "type": "TimeoutError", "message": "no result"}
        release.set()
    producer.join(timeout=10)
    consumer.join(timeout=10)

    import torch

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"producer_alloc_conf={args.producer_alloc_conf}")
    print(f"consumer_alloc_conf={args.consumer_alloc_conf}")
    print(f"warmup_consumer={args.warmup_consumer}")
    print(f"result={result}")
    print(f"producer_exit={producer.exitcode} consumer_exit={consumer.exitcode}")

    if args.expect == "pass":
        expected = result.get("status") == "pass"
    else:
        expected = result.get(
            "status"
        ) == "error" and "pidfd_getfd: Bad file descriptor" in result.get("message", "")
    return 0 if expected and producer.exitcode == 0 and consumer.exitcode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
