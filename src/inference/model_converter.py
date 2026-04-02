"""
src/inference/model_converter.py

Converts trained PyTorch / ONNX models to optimised TensorRT engine plans.

Pipeline
--------
    PyTorch model  ──torch.onnx.export──►  ONNX model
                                               │
                                      ──trt.OnnxParser──►  TRT Network
                                               │
                                      ──Builder.build──►  Serialised engine (.engine)

Usage
-----
    from src.inference.model_converter import ModelConverter

    converter = ModelConverter(
        onnx_path="model.onnx",
        engine_path="model.engine",
        fp16=True,
        workspace_mb=4096,
    )
    converter.build()

    # Or convert directly from a PyTorch module
    import torch
    converter = ModelConverter.from_torch(
        model=my_model,
        example_inputs=torch.zeros(1, 3, 224, 224),
        engine_path="model.engine",
        fp16=True,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt  # type: ignore

    TRT_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRT_AVAILABLE = False
    logger.warning("TensorRT not found – ModelConverter will not be functional.")

try:
    import importlib.util

    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


class ModelConverter:
    """
    Convert an ONNX model to an optimised TensorRT engine.

    Parameters
    ----------
    onnx_path     : path to input ONNX model
    engine_path   : destination path for the serialised TRT engine
    fp16          : enable FP16 precision (requires compatible GPU)
    int8          : enable INT8 precision (requires calibration dataset)
    workspace_mb  : max GPU workspace size in MiB (default 4096)
    min_shapes    : dict of {input_name: shape}  – for dynamic-shape opt
    opt_shapes    : dict of {input_name: shape}  – optimum shape
    max_shapes    : dict of {input_name: shape}  – maximum shape
    """

    def __init__(
        self,
        onnx_path: str,
        engine_path: str,
        *,
        fp16: bool = True,
        int8: bool = False,
        workspace_mb: int = 4096,
        min_shapes: dict[str, tuple[int, ...]] | None = None,
        opt_shapes: dict[str, tuple[int, ...]] | None = None,
        max_shapes: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.fp16 = fp16
        self.int8 = int8
        self.workspace_mb = workspace_mb
        self.min_shapes = min_shapes or {}
        self.opt_shapes = opt_shapes or {}
        self.max_shapes = max_shapes or {}

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> str:
        """
        Build and serialise the TRT engine.

        Returns
        -------
        str
            Path to the generated ``.engine`` file.
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is required to build an engine.")

        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)

        with (
            trt.Builder(trt_logger) as builder,
            builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            ) as network,
            trt.OnnxParser(network, trt_logger) as parser,
            builder.create_builder_config() as config,
        ):
            # ── Builder config ────────────────────────────────────────────
            config.max_workspace_size = self.workspace_mb * (1 << 20)

            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled.")
            elif self.fp16:
                logger.warning("FP16 requested but not supported on this GPU.")

            if self.int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled.")

            # ── Parse ONNX ────────────────────────────────────────────────
            with open(self.onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"ONNX parse failed:\n{errors}")

            logger.info("ONNX model parsed successfully.")

            # ── Dynamic-shape optimisation profiles ───────────────────────
            if self.opt_shapes:
                profile = builder.create_optimization_profile()
                for inp_name, opt_shape in self.opt_shapes.items():
                    min_shape = self.min_shapes.get(inp_name, opt_shape)
                    max_shape = self.max_shapes.get(inp_name, opt_shape)
                    profile.set_shape(inp_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)

            # ── Build serialised engine ───────────────────────────────────
            serialised = builder.build_serialized_network(network, config)
            if serialised is None:
                raise RuntimeError("TRT engine build failed.")

            os.makedirs(os.path.dirname(os.path.abspath(self.engine_path)), exist_ok=True)
            with open(self.engine_path, "wb") as f:
                f.write(serialised)

            logger.info("TRT engine saved to %s", self.engine_path)
            return self.engine_path

    # ── Convenience factory ────────────────────────────────────────────────

    @classmethod
    def from_torch(
        cls,
        model: Any,
        example_inputs: Any,
        engine_path: str,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        **kwargs: Any,
    ) -> ModelConverter:
        """
        Export a PyTorch model to ONNX and prepare a ModelConverter.

        The converter is returned without calling ``build()`` so callers
        can customise it before triggering the TRT compilation.

        Parameters
        ----------
        model         : torch.nn.Module (must be in eval mode)
        example_inputs: torch.Tensor or tuple of tensors
        engine_path   : destination path for the ``.engine`` file
        input_names   : ONNX input node names (optional)
        output_names  : ONNX output node names (optional)
        dynamic_axes  : dynamic-axis specification for ONNX export
        **kwargs      : forwarded to ModelConverter.__init__
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for from_torch().")

        import torch  # local import to keep the module importable without torch

        onnx_path = engine_path.replace(".engine", ".onnx")
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                opset_version=17,
                input_names=input_names or ["input"],
                output_names=output_names or ["output"],
                dynamic_axes=dynamic_axes,
                export_params=True,
            )

        logger.info("Model exported to ONNX: %s", onnx_path)
        return cls(onnx_path=onnx_path, engine_path=engine_path, **kwargs)

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def validate_onnx(onnx_path: str) -> bool:
        """
        Validate an ONNX model graph.

        Returns True if valid, raises on error.
        """
        try:
            import onnx  # type: ignore

            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model is valid: %s", onnx_path)
            return True
        except ImportError:
            logger.warning("onnx package not available – skipping validation.")
            return True
        except Exception as exc:
            raise ValueError(f"Invalid ONNX model: {exc}") from exc
