import argparse
import json
import os
import os.path as osp
import time

import cv2
import numpy as np
from loguru import logger

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError("onnxruntime is required. Please install it first.") from exc

from yolox.data.data_augment import preproc
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import demo_postprocess, multiclass_nms
from yolox.utils.visualize import plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack ONNXRuntime video inference")
    parser.add_argument("--video_path", type=str, required=True, help="path to input video")
    parser.add_argument("--onnx_path", type=str, required=True, help="path to onnx model")
    parser.add_argument("--output_dir", type=str, default="outputs/onnx_runs", help="output root directory")
    parser.add_argument("--score_thr", type=float, default=0.1, help="score threshold")
    parser.add_argument("--nms_thr", type=float, default=0.7, help="nms threshold")
    parser.add_argument(
        "--input_size",
        type=str,
        default="608,1088",
        help="model input size, e.g. 608,1088",
    )
    parser.add_argument("--with_p6", action="store_true", help="set for p6 model")
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider", help="ORT providers split by ','")
    # tracking args (kept close to demo_track.py)
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="frames to keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="filter out boxes with large aspect ratio",
    )
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20")
    return parser


def _parse_hw(size_str):
    h_str, w_str = size_str.split(",", 1)
    return int(h_str), int(w_str)


class ONNXPredictor:
    def __init__(self, onnx_path, input_size, with_p6=False, score_thr=0.1, nms_thr=0.7, providers=None):
        self.input_size = input_size
        self.with_p6 = with_p6
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def inference(self, frame):
        img_info = {}
        img_info["raw_img"] = frame
        img_info["height"], img_info["width"] = frame.shape[:2]

        img, ratio = preproc(frame, self.input_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        ort_inputs = {self.input_name: img[None, :, :, :]}
        outputs = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(outputs[0], self.input_size, p6=self.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        dets = multiclass_nms(
            boxes_xyxy,
            scores,
            nms_thr=self.nms_thr,
            score_thr=self.score_thr,
        )
        if dets is None:
            return None, img_info
        # Keep [x1, y1, x2, y2, score] to match BYTETracker expected format.
        return dets[:, :-1], img_info


def main():
    args = make_parser().parse_args()
    logger.info("Args: {}", args)

    if not osp.isfile(args.video_path):
        raise FileNotFoundError("video file not found: {}".format(args.video_path))
    if not osp.isfile(args.onnx_path):
        raise FileNotFoundError("onnx file not found: {}".format(args.onnx_path))

    input_size = _parse_hw(args.input_size)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    predictor = ONNXPredictor(
        onnx_path=args.onnx_path,
        input_size=input_size,
        with_p6=args.with_p6,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        providers=providers,
    )

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    run_dir = osp.join(args.output_dir, timestamp)
    metrics_dir = osp.join(run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps and src_fps > 0:
        out_fps = src_fps
    else:
        out_fps = 30.0
        logger.warning(
            "Invalid source FPS ({}) reported for video '{}'; falling back to default FPS {}. "
            "Output video timing may not match the source.",
            src_fps,
            args.video_path,
            out_fps,
        )
    output_video_path = osp.join(run_dir, osp.basename(args.video_path))
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    tracker = BYTETracker(args, frame_rate=max(1, int(round(out_fps))))
    frame_id = 0
    total_proc_time = 0.0
    per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        dets, img_info = predictor.inference(frame)

        det_count = int(dets.shape[0]) if dets is not None else 0
        online_tlwhs = []
        online_ids = []
        online_scores = []

        if dets is not None:
            online_targets = tracker.update(
                dets,
                [img_info["height"], img_info["width"]],
                [img_info["height"], img_info["width"]],
            )
            for t in online_targets:
                tlwh = t.tlwh
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(t.track_id)
                    online_scores.append(t.score)

        frame_proc_time = time.time() - frame_start
        total_proc_time += frame_proc_time
        frame_fps = 1.0 / max(1e-6, frame_proc_time)

        vis_frame = plot_tracking(
            img_info["raw_img"],
            online_tlwhs,
            online_ids,
            frame_id=frame_id + 1,
            fps=frame_fps,
        )
        writer.write(vis_frame)

        per_frame.append(
            {
                "frame_id": frame_id + 1,
                "detections": det_count,
                "valid_tracks": len(online_ids),
                "latency_ms": round(frame_proc_time * 1000.0, 3),
                "fps": round(frame_fps, 3),
            }
        )

        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps, det={}, tracks={})",
                frame_id,
                frame_fps,
                det_count,
                len(online_ids),
            )
        frame_id += 1

    cap.release()
    writer.release()

    total_frames = frame_id
    avg_latency_ms = (total_proc_time / max(1, total_frames)) * 1000.0
    avg_fps = total_frames / max(1e-6, total_proc_time)

    metrics = {
        "summary": {
            "onnx_path": args.onnx_path,
            "video_path": args.video_path,
            "providers": providers,
            "input_size": [input_size[0], input_size[1]],
            "score_thr": args.score_thr,
            "avg_fps": round(avg_fps, 4),
            "avg_latency_ms": round(avg_latency_ms, 4),
            "total_frames": int(total_frames),
            "output_video_path": output_video_path,
        },
        "per_frame": per_frame,
    }
    metrics_json_path = osp.join(metrics_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    summary_txt_path = osp.join(metrics_dir, "summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(f"avg_fps: {metrics['summary']['avg_fps']}\n")
        f.write(f"avg_latency_ms: {metrics['summary']['avg_latency_ms']}\n")
        f.write(f"total_frames: {metrics['summary']['total_frames']}\n")
        f.write(f"output_video_path: {metrics['summary']['output_video_path']}\n")
        f.write(f"metrics_json: {metrics_json_path}\n")

    logger.info("Saved video to {}", output_video_path)
    logger.info("Saved metrics json to {}", metrics_json_path)
    logger.info("Saved metrics summary to {}", summary_txt_path)


if __name__ == "__main__":
    main()
