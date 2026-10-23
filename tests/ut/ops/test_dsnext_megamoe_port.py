"""Exercise the port's host decisions without importing vLLM or an NPU runtime.

The AST loader executes the actual production functions with explicit dependency
doubles. Run this file directly with Python, or collect it with the unit suite.
These checks do not execute CANN operators or validate numerical precision.
"""

import ast
import inspect
import unittest
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions.extend(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name in names
            )
    if {node.name for node in definitions} != set(names):
        raise AssertionError("Requested production functions were not found")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace


class Quant(Enum):
    NONE = auto()
    W4A8MXFP = auto()
    W8A8MXFP = auto()
    W4A4MXFP = auto()


class Comm(Enum):
    FUSED_MC2 = auto()
    MC2 = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()


class TensorDouble:
    def __init__(self, shape, dtype="uint8", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def stride(self):
        return ()

    def view(self, dtype):
        return TensorDouble(self.shape, dtype, self.contiguous)


class TestMegaMoEPort(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size_across_dp=8),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
            kv_transfer_config=None,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=4)),
        )
        self.ascend_config = SimpleNamespace(mega_moe_max_tokens=65536, enable_fused_mc2=1)
        self.host = load_functions("vllm_ascend/ascend_config.py", ["compute_mega_moe_buffer_tokens_per_rank"], {})
        self.host.update(
            get_ascend_config=lambda: self.ascend_config,
            get_mc2_tokens_capacity=lambda: 128,
            is_mega_moe_supported=lambda: True,
            QuantType=Quant,
            MoECommType=Comm,
            A5_SUPPORT_MEGA_MOE_QUANT_TYPES={Quant.W4A8MXFP, Quant.W8A8MXFP, Quant.W4A4MXFP},
        )
        load_functions(
            "vllm_ascend/ascend_forward_context.py",
            ["get_a5_mega_moe_buffer_tokens_per_rank", "_select_capacity_and_world_size_moe_comm_method"],
            self.host,
        )
        self.shared = load_functions(
            "vllm_ascend/ops/fused_moe/mega_moe.py",
            [
                "supports_mega_moe_shared_experts",
                "_as_tensor_list",
                "_view_mxfp_scales_as_e8m0",
                "_validate_shared_mxfp_layout",
                "build_mega_moe_shared_kwargs",
            ],
            {
                "signature": inspect.signature,
                "Parameter": inspect.Parameter,
                "logger": Mock(),
                "torch": SimpleNamespace(uint8="uint8"),
                "QuantType": Quant,
                "_TORCH_FLOAT8_E8M0FNU_DTYPE": "e8m0",
                "_MEGA_MOE_SUPPORTED_SHARED_QUANTS": {Quant.W4A8MXFP, Quant.W8A8MXFP},
                "_FP4_PACK_FACTOR": 2,
                "_MXFP_SCALE_BLOCK_SIZE": 64,
                "_MXFP_SCALE_MULTIPLIER": 2,
                "_FLOAT8_E4M3FN_WEIGHT_TYPE": 24,
            },
        )

    def capacity(self):
        return self.host["get_a5_mega_moe_buffer_tokens_per_rank"](self.config, 128)

    def test_mixed_pd_uses_prefill_capacity(self):
        self.assertEqual(self.capacity(), 4096)
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_both")
        self.assertEqual(self.capacity(), 4096)

    def test_producer_uses_prefill_capacity(self):
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
        self.assertEqual(self.capacity(), 4096)

    def test_consumer_uses_decode_capacity(self):
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        self.assertEqual(self.capacity(), 128)

    def test_global_ceiling_bounds_each_ep_rank(self):
        self.ascend_config.mega_moe_max_tokens = 1024
        self.assertEqual(self.capacity(), 128)

    def test_invalid_capacities_fail_before_collectives(self):
        compute = self.host["compute_mega_moe_buffer_tokens_per_rank"]
        for ceiling, execution, ep in [(0, 128, 8), (4, 128, 8), (1024, 0, 8), (1024, 128, 0)]:
            with self.subTest(ceiling=ceiling, execution=execution, ep=ep), self.assertRaises(ValueError):
                compute(ceiling, execution, ep)

    def test_buffer_overflow_falls_back_to_existing_communication(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.assertEqual(select(4096, self.config, 128), Comm.FUSED_MC2)
        self.assertEqual(select(4097, self.config, 128), Comm.ALLTOALL)
        self.ascend_config.mega_moe_max_tokens = 512
        self.assertEqual(select(128, self.config, 128), Comm.MC2)

    def test_unquantized_draft_preserves_original_fallback(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.assertEqual(select(128, self.config, 128, True, Quant.NONE), Comm.MC2)
        self.assertEqual(select(4096, self.config, 128, True, Quant.NONE), Comm.ALLTOALL)

    def test_newer_main_quantizations_are_preserved(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        for quant in (Quant.W4A8MXFP, Quant.W8A8MXFP, Quant.W4A4MXFP):
            with self.subTest(quant=quant):
                self.assertEqual(select(128, self.config, 128, True, quant), Comm.FUSED_MC2)

    def test_shared_operator_requires_explicit_keyword_abi(self):
        def supported(
            *,
            shared_l1_weights,
            shared_l2_weights,
            shared_l1_weights_sf,
            shared_l2_weights_sf,
            shared_weight1_type,
            shared_weight2_type,
        ):
            pass

        def legacy(*args, **kwargs):
            pass

        probe = self.shared["supports_mega_moe_shared_experts"]
        self.assertTrue(probe(supported))
        self.assertFalse(probe(legacy))
        self.assertFalse(probe(object()))

    def payload(self, quant=Quant.W4A8MXFP):
        pack = 2 if quant == Quant.W4A8MXFP else 1
        return SimpleNamespace(
            hidden_states=TensorDouble((4, 128)),
            quant=SimpleNamespace(quant_type=Quant.W4A8MXFP),
            shared_weights=SimpleNamespace(
                w1=[TensorDouble((1, 128, 128 // pack))],
                w2=[TensorDouble((1, 128, 64 // pack))],
                w1_scale=[TensorDouble((1, 128, 2, 2))],
                w2_scale=[TensorDouble((1, 128, 1, 2))],
                quant_type=quant,
            ),
        )

    def test_fp8_shared_weights_have_independent_type_from_fp4_routed_weights(self):
        kwargs = self.shared["build_mega_moe_shared_kwargs"](self.payload(Quant.W8A8MXFP), 64, 296)
        self.assertEqual(kwargs["shared_weight1_type"], 24)
        self.assertEqual(kwargs["shared_weight2_type"], 24)
        self.assertEqual(kwargs["shared_l1_weights_sf"][0].dtype, "e8m0")

    def test_fp4_shared_weights_use_routed_fp4_type(self):
        kwargs = self.shared["build_mega_moe_shared_kwargs"](self.payload(), 64, 296)
        self.assertEqual(kwargs["shared_weight1_type"], 296)

    def test_incompatible_layout_fails_before_operator_call(self):
        payload = self.payload()
        payload.shared_weights.w1[0].shape = (1, 256, 64)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_noncontiguous_weights_are_rejected(self):
        payload = self.payload()
        payload.shared_weights.w1[0].contiguous = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_invalid_scale_dtype_is_rejected(self):
        payload = self.payload()
        payload.shared_weights.w2_scale[0].dtype = "float32"
        with self.assertRaisesRegex(RuntimeError, "FLOAT8_E8M0"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_missing_shared_weights_preserve_routed_only_call(self):
        payload = self.payload()
        payload.shared_weights = None
        self.assertEqual(self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296), {})


class TestFusedSharedRunner(unittest.TestCase):
    def setUp(self):
        self.input = TensorDouble((4, 128))
        self.output = TensorDouble((4, 128))
        self.weights = object()
        self.ctx = SimpleNamespace(
            moe_comm_type=Comm.FUSED_MC2,
            moe_comm_method=SimpleNamespace(supports_fused_shared_experts=lambda: True),
        )
        self.runner = SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_type=Quant.W4A8MXFP,
                dynamic_eplb=False,
                mix_placement=False,
                global_redundant_expert_num=0,
                forward_impl=Mock(return_value=(self.output, SimpleNamespace(shared_experts_fused=True))),
            ),
            ascend_shared_experts=SimpleNamespace(get_mega_moe_weights=lambda: self.weights),
            is_internal_router=False,
        )
        self.method = load_functions(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            ["_forward_with_fused_shared_experts"],
            {
                "_EXTRA_CTX": self.ctx,
                "MoECommType": Comm,
                "QuantType": Quant,
                "torch": SimpleNamespace(zeros_like=lambda value: TensorDouble(value.shape, "zeros")),
            },
        )["_forward_with_fused_shared_experts"]

    def forward(self, shared_input=None):
        return self.method(self.runner, self.input, object(), shared_input or self.input, None)

    def test_sum_is_returned_in_shared_slot_without_double_counting(self):
        shared, routed = self.forward()
        self.assertIs(shared, self.output)
        self.assertEqual(routed.shape, shared.shape)
        self.assertEqual(routed.dtype, "zeros")
        self.assertIs(self.runner.routed_experts.forward_impl.call_args.kwargs["shared_experts"], self.weights)

    def enable_cann_internal_router(self, gate):
        self.runner.is_internal_router = True
        self.runner.gate = gate
        device_gate = Mock(return_value=TensorDouble((4, 8), "float32"))
        functions = load_functions(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            ["_prepare_router_input", "_apply_router_linear", "_compute_router_logits"],
            {
                "torch": SimpleNamespace(float32="float32"),
                "DeviceOperator": SimpleNamespace(compute_gate_logits=device_gate),
            },
        )
        for name in ("_prepare_router_input", "_apply_router_linear", "_compute_router_logits"):
            setattr(self.runner, name, MethodType(functions[name], self.runner))
        return device_gate

    def test_shared_fusion_preserves_cann_gate_with_prepared_fp32_input(self):
        gate = SimpleNamespace(weight=object(), weight_fp32=object())
        device_gate = self.enable_cann_internal_router(gate)
        prepared_fp32 = TensorDouble((4, 128), "float32")
        self.method(self.runner, self.input, prepared_fp32, self.input, None)
        device_gate.assert_called_once_with(self.input, gate.weight, gate.weight_fp32, prepared_input=prepared_fp32)
        self.assertIs(
            self.runner.routed_experts.forward_impl.call_args.kwargs["router_logits"], device_gate.return_value
        )

    def test_shared_fusion_preserves_cann_gate_without_precast_weight(self):
        gate = SimpleNamespace(weight=object())
        device_gate = self.enable_cann_internal_router(gate)
        # TensorDouble has no float() method: this path must let the device
        # adaptor handle the input cast rather than allocate it in the runner.
        self.forward()
        device_gate.assert_called_once_with(self.input, gate.weight, None, prepared_input=self.input)

    def test_shared_fusion_preserves_registered_gate_without_weights(self):
        logits = TensorDouble((4, 8), "float32")
        gate = Mock(return_value=(logits, None))
        # Mock creates arbitrary attributes, so explicitly remove the optional
        # weights to model a registered gate that owns its forward operation.
        del gate.weight
        del gate.weight_fp32
        device_gate = self.enable_cann_internal_router(gate)
        self.forward()
        gate.assert_called_once_with(self.input)
        device_gate.assert_not_called()
        self.assertIs(self.runner.routed_experts.forward_impl.call_args.kwargs["router_logits"], logits)

    def test_transformed_shared_input_uses_standalone_fallback(self):
        self.assertIsNone(self.forward(TensorDouble((4, 128))))
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_legacy_operator_uses_standalone_fallback(self):
        self.ctx.moe_comm_method.supports_fused_shared_experts = lambda: False
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_incompatible_shared_weights_use_standalone_fallback(self):
        self.runner.ascend_shared_experts.get_mega_moe_weights = lambda: None
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_other_routed_quantizations_keep_existing_path(self):
        self.runner.routed_experts.quant_type = Quant.W8A8MXFP
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_dynamic_eplb_keeps_existing_path(self):
        self.runner.routed_experts.dynamic_eplb = True
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_redundant_experts_keep_existing_path(self):
        self.runner.routed_experts.global_redundant_expert_num = 1
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_missing_fusion_marker_fails_instead_of_losing_shared_output(self):
        self.runner.routed_experts.forward_impl.return_value = self.output, SimpleNamespace(shared_experts_fused=False)
        with self.assertRaisesRegex(RuntimeError, "without fusing"):
            self.forward()


class TestSharedWeightLifecycle(unittest.TestCase):
    def setUp(self):
        self.shared = SimpleNamespace(
            quant_type=Quant.W4A8MXFP,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=0.0,
            lora_context=False,
            weights_replicated=False,
            _mega_moe_weights=None,
            moe_config=SimpleNamespace(
                hidden_dim=128,
                intermediate_size_per_partition=64,
                tp_group=SimpleNamespace(world_size=1),
                in_dtype="bf16",
                swiglu_limit=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                is_sequence_parallel=False,
            ),
            layer=SimpleNamespace(
                act_fn=type("SiluAndMul", (), {})(),
                gate_up_proj=SimpleNamespace(weight=TensorDouble((64, 256)), weight_scale=TensorDouble((2, 256, 2))),
                down_proj=SimpleNamespace(weight=TensorDouble((32, 128)), weight_scale=TensorDouble((1, 128, 2))),
            ),
        )
        self.functions = load_functions(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            ["__init__", "get_mega_moe_weights"],
            {
                "QuantType": Quant,
                "has_lora": bool,
                "wraps": wraps,
                "torch": SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"),
                "SituAndMul": type("SituAndMul", (), {}),
                "get_ascend_config": lambda: SimpleNamespace(
                    multistream_overlap_shared_expert=False, enable_shared_expert_dp=False
                ),
            },
        )

    def test_weight_reload_invalidates_contiguous_shared_cache(self):
        calls = []
        quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: calls.append(layer))
        self.functions["__init__"](self.shared, self.shared.layer, self.shared.moe_config, Quant.W4A8MXFP, quant_method)
        self.shared._mega_moe_weights = object()
        quant_method.process_weights_after_loading(self.shared.layer)
        self.assertIsNone(self.shared._mega_moe_weights)
        self.assertEqual(calls, [self.shared.layer])

    def test_active_lora_cannot_reuse_cached_fused_shared_weights(self):
        self.shared._mega_moe_weights = object()
        self.shared.lora_context = True
        self.assertIsNone(self.functions["get_mega_moe_weights"](self.shared))

    def test_incompatible_shared_layout_falls_back_before_contiguous_copy(self):
        # TensorDouble deliberately has no transpose/contiguous allocation API.
        self.assertIsNone(self.functions["get_mega_moe_weights"](self.shared))


class TestMegaMoEConfigLifetime(unittest.TestCase):
    def setUp(self):
        path = "vllm_ascend/ops/fused_moe/moe_comm_method.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FusedMC2CommImpl")
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"__init__", "_init_mega_moe_symm_buffer", "_apply_cann_mega_moe"}
        ]

        class Dispatcher:
            global_bs = 0
            max_num_tokens_per_rank = 128
            ep_world_size = 8
            ep_rank_id = 0

        class Base:
            def __init__(self, config):
                self.moe_config = config
                self.token_dispatcher = Dispatcher()

        self.config_getter = Mock(return_value=object())
        self.capacity = Mock(return_value=4)
        self.allocate = Mock(return_value=SimpleNamespace())
        self.op = Mock(return_value=(object(), object()))
        self.ascend_config = SimpleNamespace(enable_fused_mc2=1, mega_moe_max_tokens=65536)
        self.ns = {
            "MoECommMethod": Base,
            "TokenDispatcherWithMC2": Dispatcher,
            "get_ascend_config": lambda: self.ascend_config,
            "is_mega_moe_supported": lambda: True,
            "get_current_vllm_config": self.config_getter,
            "get_a5_mega_moe_buffer_tokens_per_rank": self.capacity,
            "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
            "HardwareCapability": SimpleNamespace(CANN_MEGAMOE_MXFP=object()),
            "get_mc2_group": lambda: SimpleNamespace(device_group=object()),
            "logger": Mock(),
            "cast": lambda _, value: value,
            "torch": SimpleNamespace(zeros=Mock(), int32="int32", float32="float32", nn=SimpleNamespace(Module=object)),
            "moe_utils": SimpleNamespace(
                load_cann_mega_moe_ops=lambda: (self.allocate, self.op),
                select_mega_moe_activation_kwargs=lambda *args, **kwargs: {},
                _get_cann_mega_moe_quant_settings=lambda _: (4, 24, 296),
            ),
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
        exec(compile(isolated, path, "exec"), self.ns)
        config = SimpleNamespace(
            activation="silu",
            swiglu_limit=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
            num_local_experts=1,
            num_experts=8,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size_per_partition=64,
        )
        self.comm = self.ns["FusedMC2CommImpl"](config)
        # Model loading has finished; V2 CustomOp/profile execution has no
        # current config. Both the capacity helper and getter are forbidden.
        self.config_getter.side_effect = AssertionError("Current vLLM config is not set")
        self.capacity.side_effect = AssertionError("capacity must be resolved during initialization")

    def apply(self, tokens):
        payload = SimpleNamespace(
            hidden_states=TensorDouble((tokens, 128)),
            topk_ids=Mock(),
            topk_weights=Mock(),
            quant=SimpleNamespace(quant_type=Quant.W4A8MXFP),
            layer=SimpleNamespace(mega_moe_activation_kwargs={}),
            shared_weights=None,
        )
        weights = SimpleNamespace(w1=[], w2=[], w1_scale=[], w2_scale=[], w1_scale_bias=None, w2_scale_bias=None)
        return self.comm._apply_cann_mega_moe(payload, weights, is_decode_only_node=False)

    def test_first_forward_allocates_buffer_without_current_config(self):
        result = self.apply(4)
        self.assertEqual(result, self.op.return_value)
        self.assertEqual(self.allocate.call_args.args[2], 4)
        self.assertEqual(self.config_getter.call_count, 1)
        self.assertEqual(self.capacity.call_count, 1)

    def test_later_forward_reuses_buffer_without_current_config(self):
        self.apply(4)
        self.apply(2)
        self.allocate.assert_called_once()
        self.assertEqual(self.op.call_count, 2)
        self.assertEqual(self.config_getter.call_count, 1)

    def test_overflow_rejected_before_allocation_or_operator(self):
        with self.assertRaisesRegex(ValueError, "received 5 tokens"):
            self.apply(5)
        self.allocate.assert_not_called()
        self.op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
