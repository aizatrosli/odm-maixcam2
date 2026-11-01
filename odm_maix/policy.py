"""Walking-policy backends.

* ``OnnxPolicy`` runs the original ONNX file on the A53 CPU (onnxruntime).
* ``NpuPolicy`` runs the Pulsar2-compiled ``.axmodel`` on the AX630C NPU
  through ``maix.nn.NN`` (load the ``.mud`` file that points at it).

Both take the 101-float observation and return 14 float actions, so the
control loop does not care which one it gets.
"""

from __future__ import annotations

import numpy as np

OBS_SIZE = 101
ACTION_SIZE = 14


class OnnxPolicy:
    def __init__(self, path: str, input_name: str = "obs"):
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1  # leave the second A53 core for I/O threads
        self.sess = onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input_name = input_name

    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, OBS_SIZE)
        return self.sess.run(None, {self.input_name: x})[0][0]


class NpuPolicy:
    def __init__(self, mud_path: str, input_name: str = "obs"):
        from maix import nn, tensor

        self._tensor = tensor
        self.model = nn.NN(mud_path)
        self.input_name = input_name
        try:
            outs = self.model.outputs_info()
            self.output_name = outs[0].name
        except Exception:
            self.output_name = None

    def infer(self, obs: np.ndarray) -> np.ndarray:
        t = self._tensor
        x = np.asarray(obs, dtype=np.float32).reshape(1, OBS_SIZE)
        inputs = t.Tensors()
        inputs.add_tensor(self.input_name, t.tensor_from_numpy_float32(x), True, True)
        outputs = self.model.forward(inputs)
        key = self.output_name or outputs.keys()[0]
        y = t.tensor_to_numpy_float32(outputs[key])
        return np.asarray(y, dtype=np.float32).reshape(-1)[:ACTION_SIZE]


def load_policy(backend: str, path: str):
    backend = backend.lower()
    if backend == "onnx":
        return OnnxPolicy(path)
    if backend == "npu":
        return NpuPolicy(path)
    raise ValueError(f"unknown policy backend {backend!r} (use 'onnx' or 'npu')")
