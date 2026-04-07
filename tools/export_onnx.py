from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module

import argparse
import os


def _parse_input_size(value):
    # Accept "H,W" or "H"
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if "," in value:
        h_str, w_str = value.split(",", 1)
        return (int(h_str), int(w_str))
    side = int(value)
    return (side, side)


def _format_onnx_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(dim.dim_value)
        else:
            dims.append("?")
    return dims


def _log_onnx_io_info(onnx_path):
    try:
        import onnx
    except ImportError:
        logger.warning("onnx is not installed, skip printing ONNX tensor info.")
        return

    model = onnx.load(onnx_path)
    logger.info("ONNX IO summary:")
    for tensor in model.graph.input:
        logger.info(
            "  input  name={} dtype={} shape={}".format(
                tensor.name, tensor.type.tensor_type.elem_type, _format_onnx_shape(tensor)
            )
        )
    for tensor in model.graph.output:
        logger.info(
            "  output name={} dtype={} shape={}".format(
                tensor.name, tensor.type.tensor_type.elem_type, _format_onnx_shape(tensor)
            )
        )


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="bytetrack_s.onnx", help="(deprecated) output model path"
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="output path of exported onnx model"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output-node", "--output", dest="output_node", default="output", type=str,
        help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default=None,
        help="model input size, e.g. 608,1088 or 640",
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-c", "--ckpt", "--weights", dest="ckpt", default=None, type=str, help="ckpt/weights path"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_path = args.output_path or args.output_name
    input_size = _parse_input_size(args.input_size) if args.input_size else tuple(exp.test_size)

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx._export(
        model,
        dummy_input,
        output_path,
        input_names=[args.input],
        output_names=[args.output_node],
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(output_path))
    _log_onnx_io_info(output_path)

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_path)
        logger.info("generated simplified onnx model named {}".format(output_path))
        _log_onnx_io_info(output_path)


if __name__ == "__main__":
    main()
