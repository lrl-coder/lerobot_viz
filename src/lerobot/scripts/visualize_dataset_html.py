#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from threading import Lock

import av
import numpy as np
import pandas as pd
import requests
from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from PIL import Image as PILImage

from lerobot import available_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import IterableNamespace
from lerobot.utils.utils import init_logging

DEPTH_KEY_PREFIX = "observation.images.depth."
DEPTH_COLORMAP = np.array(
    [
        [31, 12, 72],
        [38, 70, 180],
        [25, 180, 220],
        [60, 210, 95],
        [245, 220, 45],
        [235, 55, 35],
    ],
    dtype=np.float32,
)


def _require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "Depth visualization requires h5py. Install it with uv pip install h5py."
        ) from error
    return h5py


def get_depth_episode_path(dataset_root: Path, episode_id: int, depth_dir: Path | None = None) -> Path:
    depth_dir = Path(depth_dir) if depth_dir is not None else Path(dataset_root) / "depth"
    return depth_dir / f"episode_{episode_id:06d}.h5"


def get_depth_keys(depth_path: Path) -> list[str]:
    """Return supported frame-first depth datasets from an episode HDF5 file."""
    h5py = _require_h5py()
    depth_keys = []
    with h5py.File(depth_path, "r") as depth_file:

        def collect_depth_dataset(name, value):
            if (
                isinstance(value, h5py.Dataset)
                and name.startswith(DEPTH_KEY_PREFIX)
                and value.ndim == 3
                and np.issubdtype(value.dtype, np.number)
            ):
                depth_keys.append(name)

        depth_file.visititems(collect_depth_dataset)
    return sorted(depth_keys)


def get_rgb_key_for_depth(depth_key: str, rgb_keys: list[str]) -> str | None:
    """Match observation.images.depth.<camera> to the RGB stream for the same camera."""
    camera_name = depth_key.removeprefix(DEPTH_KEY_PREFIX)
    preferred_key = f"observation.images.{camera_name}"
    if preferred_key in rgb_keys:
        return preferred_key

    suffix = f".{camera_name}"
    return next((key for key in rgb_keys if key.endswith(suffix)), None)


def insert_depth_videos_next_to_rgb(videos_info: list[dict], depth_videos_info: list[dict]) -> list[dict]:
    """Insert each depth video immediately after its matching RGB video."""
    depth_by_rgb = {info.get("rgb_key"): info for info in depth_videos_info if info.get("rgb_key")}
    ordered_videos = []
    inserted_depth_keys = set()
    for video_info in videos_info:
        ordered_videos.append(video_info)
        depth_info = depth_by_rgb.get(video_info.get("camera_key"))
        if depth_info is not None:
            ordered_videos.append(depth_info)
            inserted_depth_keys.add(depth_info["camera_key"])

    ordered_videos.extend(info for info in depth_videos_info if info["camera_key"] not in inserted_depth_keys)
    return ordered_videos


def get_depth_range(depth_dataset, max_sample_frames: int = 32) -> tuple[float, float]:
    """Compute a stable visualization range from sampled, non-zero depth pixels."""
    frame_count = len(depth_dataset)
    if frame_count == 0:
        raise ValueError("Depth dataset contains no frames.")

    sample_indices = np.unique(
        np.linspace(0, frame_count - 1, min(frame_count, max_sample_frames), dtype=int)
    )
    sampled_depth = np.asarray(depth_dataset[sample_indices], dtype=np.float32)
    valid_depth = sampled_depth[np.isfinite(sampled_depth) & (sampled_depth > 0)]
    if valid_depth.size == 0:
        return 0.0, 1.0

    depth_min, depth_max = np.percentile(valid_depth, (1, 99))
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    return float(depth_min), float(depth_max)


def colorize_depth_frame(depth_frame: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Map a scalar depth frame to RGB; zero and non-finite pixels remain black."""
    depth_frame = np.asarray(depth_frame, dtype=np.float32)
    valid_mask = np.isfinite(depth_frame) & (depth_frame > 0)
    normalized = (depth_frame - depth_min) / (depth_max - depth_min)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)

    color_positions = normalized * (len(DEPTH_COLORMAP) - 1)
    lower_indices = np.floor(color_positions).astype(np.intp)
    upper_indices = np.minimum(lower_indices + 1, len(DEPTH_COLORMAP) - 1)
    blend = (color_positions - lower_indices)[..., None]
    colored = DEPTH_COLORMAP[lower_indices] * (1.0 - blend) + DEPTH_COLORMAP[upper_indices] * blend
    colored[~valid_mask] = 0
    return colored.astype(np.uint8)


def run_server(
    dataset: LeRobotDataset | IterableNamespace | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
    depth_dir: Path | None = None,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    video_generation_locks: dict[Path, Lock] = {}
    video_generation_locks_guard = Lock()

    def get_episode_position(episode_id: int) -> int:
        if dataset.episodes is not None:
            try:
                return dataset.episodes.index(episode_id)
            except ValueError:
                abort(404)
        if episode_id < 0 or episode_id >= len(dataset.episode_data_index["from"]):
            abort(404)
        return episode_id

    def get_image_video_path(episode_id: int, image_key: str) -> Path:
        safe_image_key = re.sub(r"[^A-Za-z0-9_.-]", "_", image_key)
        chunk = episode_id // dataset.meta.chunks_size
        return (
            dataset.root
            / ".lerobot_viz_cache"
            / "videos"
            / f"chunk-{chunk:03d}"
            / safe_image_key
            / f"episode_{episode_id:06d}.mp4"
        )

    def get_depth_video_path(episode_id: int, depth_key: str) -> Path:
        safe_depth_key = re.sub(r"[^A-Za-z0-9_.-]", "_", depth_key)
        chunk = episode_id // dataset.meta.chunks_size
        return (
            dataset.root
            / ".lerobot_viz_cache"
            / "depth-videos"
            / f"chunk-{chunk:03d}"
            / safe_depth_key
            / f"episode_{episode_id:06d}.mp4"
        )

    def encode_video_with_ffmpeg(
        image_column,
        from_idx: int,
        to_idx: int,
        video_path: Path,
    ) -> tuple[bool, str]:
        command = [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            str(dataset.fps),
            "-vcodec",
            "png",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
        ]
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(dataset.fps),
                "-movflags",
                "+faststart",
                str(video_path),
            ]
        )

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame_index in range(from_idx, to_idx):
                image_data = image_column[frame_index].as_py()
                process.stdin.write(image_data["bytes"])
            process.stdin.close()
        except BrokenPipeError:
            pass

        error = process.stderr.read().decode(errors="replace")
        return process.wait() == 0, error

    def encode_video_with_pyav(
        image_column,
        from_idx: int,
        to_idx: int,
        video_path: Path,
    ) -> None:
        first_image = image_column[from_idx].as_py()
        with PILImage.open(BytesIO(first_image["bytes"])) as image:
            width, height = image.size

        with av.open(
            str(video_path),
            mode="w",
            format="mp4",
            options={"movflags": "faststart"},
        ) as output:
            stream = output.add_stream(
                "libx264",
                dataset.fps,
                options={
                    "crf": "23",
                    "preset": "veryfast",
                    "g": str(dataset.fps),
                },
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for frame_index in range(from_idx, to_idx):
                image_data = image_column[frame_index].as_py()
                with PILImage.open(BytesIO(image_data["bytes"])) as image:
                    video_frame = av.VideoFrame.from_image(image.convert("RGB"))
                for packet in stream.encode(video_frame):
                    output.mux(packet)

            for packet in stream.encode():
                output.mux(packet)

    def generate_image_video(episode_id: int, image_key: str) -> Path:
        video_path = get_image_video_path(episode_id, image_key)
        if video_path.is_file():
            return video_path

        with video_generation_locks_guard:
            generation_lock = video_generation_locks.setdefault(video_path, Lock())

        with generation_lock:
            if video_path.is_file():
                return video_path

            episode_position = get_episode_position(episode_id)
            from_idx = int(dataset.episode_data_index["from"][episode_position])
            to_idx = int(dataset.episode_data_index["to"][episode_position])
            image_column = dataset.hf_dataset.data.column(image_key)

            video_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = video_path.with_suffix(".tmp.mp4")
            logging.info(
                "Generating cached H.264 video for episode %s, %s (%s frames)",
                episode_id,
                image_key,
                to_idx - from_idx,
            )
            try:
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path is not None:
                    succeeded, error = encode_video_with_ffmpeg(
                        image_column,
                        from_idx,
                        to_idx,
                        temporary_path,
                    )
                    if not succeeded:
                        raise RuntimeError(f"FFmpeg video encoding failed:\n{error.strip()}")
                    logging.info("Encoded %s with FFmpeg libx264", image_key)
                else:
                    logging.warning("FFmpeg not found; falling back to PyAV for H.264 encoding")
                    encode_video_with_pyav(image_column, from_idx, to_idx, temporary_path)

                temporary_path.replace(video_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

            logging.info("Cached video ready: %s", video_path)
            return video_path

    def encode_depth_video_with_ffmpeg(
        depth_dataset,
        depth_min: float,
        depth_max: float,
        video_path: Path,
    ) -> tuple[bool, str]:
        _, height, width = depth_dataset.shape
        command = [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(dataset.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(dataset.fps),
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for depth_frame in depth_dataset:
                process.stdin.write(colorize_depth_frame(depth_frame, depth_min, depth_max).tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass

        error = process.stderr.read().decode(errors="replace")
        return process.wait() == 0, error

    def encode_depth_video_with_pyav(
        depth_dataset,
        depth_min: float,
        depth_max: float,
        video_path: Path,
    ) -> None:
        _, height, width = depth_dataset.shape
        with av.open(str(video_path), mode="w", format="mp4", options={"movflags": "faststart"}) as output:
            stream = output.add_stream(
                "libx264",
                dataset.fps,
                options={"crf": "18", "preset": "veryfast", "g": str(dataset.fps)},
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            for depth_frame in depth_dataset:
                color_frame = colorize_depth_frame(depth_frame, depth_min, depth_max)
                video_frame = av.VideoFrame.from_ndarray(color_frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    output.mux(packet)

            for packet in stream.encode():
                output.mux(packet)

    def generate_depth_video(episode_id: int, depth_key: str) -> Path:
        depth_path = get_depth_episode_path(dataset.root, episode_id, depth_dir)
        video_path = get_depth_video_path(episode_id, depth_key)
        if video_path.is_file() and video_path.stat().st_mtime >= depth_path.stat().st_mtime:
            return video_path

        with video_generation_locks_guard:
            generation_lock = video_generation_locks.setdefault(video_path, Lock())

        with generation_lock:
            if video_path.is_file() and video_path.stat().st_mtime >= depth_path.stat().st_mtime:
                return video_path

            h5py = _require_h5py()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = video_path.with_suffix(".tmp.mp4")
            try:
                with h5py.File(depth_path, "r") as depth_file:
                    if depth_key not in depth_file:
                        abort(404)
                    depth_dataset = depth_file[depth_key]
                    episode_position = get_episode_position(episode_id)
                    expected_frames = int(
                        dataset.episode_data_index["to"][episode_position]
                        - dataset.episode_data_index["from"][episode_position]
                    )
                    if len(depth_dataset) != expected_frames:
                        raise ValueError(
                            f"Depth stream {depth_key!r} has {len(depth_dataset)} frames, "
                            f"but episode {episode_id} has {expected_frames} RGB frames."
                        )

                    depth_min, depth_max = get_depth_range(depth_dataset)
                    logging.info(
                        "Generating cached depth video for episode %s, %s (%s frames, range %.1f..%.1f)",
                        episode_id,
                        depth_key,
                        len(depth_dataset),
                        depth_min,
                        depth_max,
                    )
                    if shutil.which("ffmpeg") is not None:
                        succeeded, error = encode_depth_video_with_ffmpeg(
                            depth_dataset, depth_min, depth_max, temporary_path
                        )
                        if not succeeded:
                            raise RuntimeError(f"FFmpeg depth video encoding failed:\n{error.strip()}")
                    else:
                        logging.warning("FFmpeg not found; falling back to PyAV for depth video encoding")
                        encode_depth_video_with_pyav(depth_dataset, depth_min, depth_max, temporary_path)

                temporary_path.replace(video_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

            logging.info("Cached depth video ready: %s", video_path)
            return video_path

    @app.route("/")
    def hommepage(dataset=dataset):
        if dataset:
            dataset_namespace, dataset_name = dataset.repo_id.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param = None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])

        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route(
        "/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>/image-video/<path:image_key>"
    )
    def show_image_video(
        dataset_namespace,
        dataset_name,
        episode_id,
        image_key,
        dataset=dataset,
    ):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        if (
            not isinstance(dataset, LeRobotDataset)
            or dataset.repo_id != repo_id
            or image_key not in dataset.meta.image_keys
        ):
            abort(404)

        video_path = generate_image_video(episode_id, image_key)
        return send_file(
            video_path,
            mimetype="video/mp4",
            conditional=True,
            max_age=31536000,
        )

    @app.route(
        "/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>/depth-video/<path:depth_key>"
    )
    def show_depth_video(
        dataset_namespace,
        dataset_name,
        episode_id,
        depth_key,
        dataset=dataset,
    ):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        if not isinstance(dataset, LeRobotDataset) or dataset.repo_id != repo_id:
            abort(404)

        depth_path = get_depth_episode_path(dataset.root, episode_id, depth_dir)
        if not depth_path.is_file() or depth_key not in get_depth_keys(depth_path):
            abort(404)

        video_path = generate_depth_video(episode_id, depth_key)
        return send_file(
            video_path,
            mimetype="video/mp4",
            conditional=True,
            max_age=31536000,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        try:
            if dataset is None:
                dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. See how to convert your dataset at https://github.com/huggingface/lerobot/pull/461",
                400,
            )
        dataset_version = (
            str(dataset.meta._version) if isinstance(dataset, LeRobotDataset) else dataset.codebase_version
        )
        match = re.search(r"v(\d+)\.", dataset_version)
        if match:
            major_version = int(match.group(1))
            if major_version < 2:
                return "Make sure to convert your LeRobotDataset to v2 & above."

        episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
        dataset_info = {
            "repo_id": f"{dataset_namespace}/{dataset_name}",
            "num_samples": dataset.num_frames
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_frames,
            "num_episodes": dataset.num_episodes
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_episodes,
            "fps": dataset.fps,
        }
        has_generated_videos = False
        check_video_codec = False
        if isinstance(dataset, LeRobotDataset):
            video_paths = [
                dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
            ]
            videos_info = [
                {
                    "url": url_for("static", filename=str(video_path).replace("\\", "/")),
                    "filename": video_path.parent.name,
                    "camera_key": key,
                    "generated": False,
                }
                for key, video_path in zip(dataset.meta.video_keys, video_paths, strict=True)
            ]
            check_video_codec = bool(video_paths)
            for image_key in dataset.meta.image_keys:
                videos_info.append(
                    {
                        "url": url_for(
                            "show_image_video",
                            dataset_namespace=dataset_namespace,
                            dataset_name=dataset_name,
                            episode_id=episode_id,
                            image_key=image_key,
                        ),
                        "filename": image_key,
                        "camera_key": image_key,
                        "generated": True,
                    }
                )

            depth_path = get_depth_episode_path(dataset.root, episode_id, depth_dir)
            depth_videos_info = []
            if depth_path.is_file():
                rgb_keys = list(dataset.meta.camera_keys)
                for depth_key in get_depth_keys(depth_path):
                    depth_videos_info.append(
                        {
                            "url": url_for(
                                "show_depth_video",
                                dataset_namespace=dataset_namespace,
                                dataset_name=dataset_name,
                                episode_id=episode_id,
                                depth_key=depth_key,
                            ),
                            "filename": depth_key,
                            "camera_key": depth_key,
                            "rgb_key": get_rgb_key_for_depth(depth_key, rgb_keys),
                            "generated": True,
                            "is_depth": True,
                        }
                    )
                videos_info = insert_depth_videos_next_to_rgb(videos_info, depth_videos_info)

            has_generated_videos = any(info["generated"] for info in videos_info)
            tasks = dataset.meta.episodes[episode_id]["tasks"]
        else:
            video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
            videos_info = [
                {
                    "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                    + dataset.video_path.format(
                        episode_chunk=int(episode_id) // dataset.chunks_size,
                        video_key=video_key,
                        episode_index=episode_id,
                    ),
                    "filename": video_key,
                    "generated": False,
                }
                for video_key in video_keys
            ]
            check_video_codec = bool(video_keys)

            response = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl", timeout=5
            )
            response.raise_for_status()
            # Split into lines and parse each line as JSON
            tasks_jsonl = [json.loads(line) for line in response.text.splitlines() if line.strip()]

            filtered_tasks_jsonl = [row for row in tasks_jsonl if row["episode_index"] == episode_id]
            tasks = filtered_tasks_jsonl[0]["tasks"]

        if episodes is None:
            episodes = list(
                range(dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes)
            )

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            has_generated_videos=has_generated_videos,
            check_video_codec=check_video_codec,
            language_instruction=tasks,
            episode_data_csv_str=episode_data_csv_str,
            columns=columns,
            ignored_columns=ignored_columns,
        )

    app.run(host=host, port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def get_episode_data(dataset: LeRobotDataset | IterableNamespace, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    selected_columns = []
    for column_name, feature in dataset.features.items():
        try:
            is_numeric = np.issubdtype(np.dtype(feature["dtype"]), np.number)
        except TypeError:
            is_numeric = False
        if is_numeric and column_name != "timestamp":
            selected_columns.append(column_name)

    ignored_columns = []
    supported_columns = []
    for column_name in selected_columns:
        if len(dataset.features[column_name]["shape"]) > 1:
            ignored_columns.append(column_name)
        else:
            supported_columns.append(column_name)
    selected_columns = supported_columns

    # init header of csv with state and action names
    header = ["timestamp"]

    for column_name in selected_columns:
        dim_state = (
            dataset.meta.shapes[column_name][0]
            if isinstance(dataset, LeRobotDataset)
            else dataset.features[column_name].shape[0]
        )

        feature = dataset.features[column_name]
        column_names = feature["names"] if "names" in feature else None
        if not isinstance(column_names, list) or len(column_names) != dim_state:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    if isinstance(dataset, LeRobotDataset):
        episode_position = (
            dataset.episodes.index(episode_index) if dataset.episodes is not None else episode_index
        )
        from_idx = dataset.episode_data_index["from"][episode_position]
        to_idx = dataset.episode_data_index["to"][episode_position]
        data = (
            dataset.hf_dataset.select(range(from_idx, to_idx))
            .select_columns(selected_columns)
            .with_format("pandas")
        )
    else:
        repo_id = dataset.repo_id

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
            episode_chunk=int(episode_index) // dataset.chunks_size, episode_index=episode_index
        )
        df = pd.read_parquet(url)
        data = df[selected_columns]  # Select specific columns

    rows = np.hstack(
        (
            np.expand_dims(data["timestamp"], axis=1),
            *[np.vstack(data[col]) for col in selected_columns[1:]],
        )
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.features:
        return None

    # get first frame index
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


def get_dataset_info(repo_id: str) -> IterableNamespace:
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    depth_dir: Path | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
                depth_dir=None,
            )
    else:
        # Create a simlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset) and dataset.meta.video_keys:
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir, depth_dir=depth_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing episode_XXXXXX.h5 depth files. "
            "By default, automatically uses ROOT/depth when it exists."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")
    episodes = kwargs.get("episodes")

    dataset = None
    if repo_id:
        dataset = (
            LeRobotDataset(repo_id, root=root, episodes=episodes, tolerance_s=tolerance_s)
            if not load_from_hf_hub
            else get_dataset_info(repo_id)
        )

    visualize_dataset_html(dataset, **vars(args))


if __name__ == "__main__":
    main()
