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

import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from flask import Flask, render_template

from lerobot.datasets.utils import IterableNamespace
from lerobot.scripts.visualize_dataset_html import (
    colorize_depth_frame,
    get_depth_episode_path,
    get_depth_keys,
    get_episode_data,
    get_raw_image_frame_paths,
    get_rgb_key_for_depth,
    insert_depth_videos_next_to_rgb,
    visualize_dataset_html,
)


def test_visualize_dataset_html_clears_only_generated_video_cache_on_start(tmp_path):
    output_dir = tmp_path / "output"
    generated_video = output_dir / "static" / "generated-videos" / "depth" / "stale.mp4"
    generated_video.parent.mkdir(parents=True)
    generated_video.touch()
    preserved_file = output_dir / "static" / "keep.txt"
    preserved_file.touch()

    visualize_dataset_html(dataset=None, output_dir=output_dir, serve=False)

    assert not generated_video.parent.parent.exists()
    assert preserved_file.is_file()


def test_get_depth_episode_path_defaults_to_video_depth_and_allows_override(tmp_path):
    assert get_depth_episode_path(tmp_path, episode_id=3) == (
        tmp_path / "video" / "depth" / "episode_000003.h5"
    )

    custom_depth_dir = tmp_path / "custom-depth"
    assert get_depth_episode_path(tmp_path, episode_id=4, depth_dir=custom_depth_dir) == (
        custom_depth_dir / "episode_000004.h5"
    )


def test_get_raw_image_frame_paths_supports_capture_layout_and_numeric_sort(tmp_path):
    image_dir = (
        tmp_path
        / "video"
        / "images"
        / "observation.images.wrist"
        / "episode_000003"
    )
    image_dir.mkdir(parents=True)
    for filename in ("frame_000010.png", "frame_000002.png", "frame_000001.png"):
        (image_dir / filename).touch()

    frame_paths = get_raw_image_frame_paths(
        tmp_path,
        episode_id=3,
        camera_key="observation.images.wrist",
    )

    assert [path.name for path in frame_paths] == [
        "frame_000001.png",
        "frame_000002.png",
        "frame_000010.png",
    ]


def test_get_episode_data_supports_all_numeric_dtypes(monkeypatch):
    dataset = IterableNamespace(
        {
            "repo_id": "local/no_video",
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                "action": {"dtype": "float64", "shape": [2], "names": ["x"]},
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["position", "velocity"],
                },
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "observation.image": {
                    "dtype": "image",
                    "shape": [8, 8, 3],
                    "names": ["height", "width", "channels"],
                },
            },
        }
    )
    frame = pd.DataFrame(
        {
            "action": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "observation.state": [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            "frame_index": [0, 1],
            "timestamp": [0.0, 0.1],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)

    csv_text, columns, ignored_columns = get_episode_data(dataset, 0)
    rows = list(csv.reader(StringIO(csv_text)))

    assert [column["key"] for column in columns] == [
        "action",
        "observation.state",
        "frame_index",
    ]
    assert columns[0]["value"] == ["action_0", "action_1"]
    assert ignored_columns == []
    assert len(rows[0]) == len(rows[1]) == 6


def test_template_renders_generated_video_for_image_dataset():
    template_folder = Path(__file__).parents[2] / "src" / "lerobot" / "templates"
    app = Flask(__name__, template_folder=template_folder)

    with app.app_context():
        page = render_template(
            "visualize_dataset_template.html",
            episode_id=0,
            episodes=[0],
            dataset_info={
                "repo_id": "local/no_video",
                "num_samples": 2,
                "num_episodes": 1,
                "fps": 10,
            },
            videos_info=[
                {
                    "url": "/local/no_video/episode_0/image-video/observation.image",
                    "filename": "observation.image",
                    "generated": True,
                }
            ],
            has_generated_videos=True,
            check_video_codec=False,
            language_instruction=["Do the task."],
            episode_data_csv_str="timestamp,state_0\r\n0.0,1.0\r\n",
            columns=[{"key": "state", "value": ["state_0"]}],
            ignored_columns=[],
        )

    assert "<video" in page
    assert "filter videos" in page
    assert "/local/no_video/episode_0/image-video/observation.image" in page
    assert "generates and caches H.264 videos" in page
    assert "Language Instruction:" in page
    assert "Do the task." in page
    assert "nVideos: 1" in page
    assert "grid grid-cols-4" in page
    assert "x-show='!videoCodecError && videosKeysSelected.includes(\"observation.image\")'" in page


def test_depth_streams_are_matched_and_placed_next_to_rgb():
    rgb_videos = [
        {"camera_key": "observation.images.third_view"},
        {"camera_key": "observation.images.wrist"},
    ]
    depth_videos = [
        {
            "camera_key": "observation.images.depth.wrist",
            "rgb_key": "observation.images.wrist",
        },
        {
            "camera_key": "observation.images.depth.third_view",
            "rgb_key": "observation.images.third_view",
        },
    ]

    ordered = insert_depth_videos_next_to_rgb(rgb_videos, depth_videos)

    assert [video["camera_key"] for video in ordered] == [
        "observation.images.third_view",
        "observation.images.depth.third_view",
        "observation.images.wrist",
        "observation.images.depth.wrist",
    ]
    assert (
        get_rgb_key_for_depth(
            "observation.images.depth.third_view",
            [video["camera_key"] for video in rgb_videos],
        )
        == "observation.images.third_view"
    )


def test_colorize_depth_frame_uses_black_for_invalid_depth():
    depth = np.array([[0, 100, 200], [np.nan, 150, np.inf]], dtype=np.float32)

    colored = colorize_depth_frame(depth, depth_min=100, depth_max=200)

    assert colored.shape == (2, 3, 3)
    assert colored.dtype == np.uint8
    assert np.all(colored[0, 0] == 0)
    assert np.all(colored[1, 0] == 0)
    assert np.all(colored[1, 2] == 0)
    assert not np.array_equal(colored[0, 1], colored[0, 2])


def test_get_depth_keys_filters_non_depth_datasets(tmp_path):
    h5py = pytest.importorskip("h5py")
    depth_path = tmp_path / "episode_000000.h5"
    with h5py.File(depth_path, "w") as depth_file:
        depth_file.create_dataset(
            "observation.images.depth.third_view",
            data=np.zeros((2, 4, 5), dtype=np.uint16),
        )
        depth_file.create_dataset(
            "observation.images.depth.invalid_shape",
            data=np.zeros((2, 4), dtype=np.uint16),
        )
        depth_file.create_dataset("camera_timestamp_ns", data=np.zeros(2, dtype=np.int64))

    assert get_depth_keys(depth_path) == ["observation.images.depth.third_view"]
