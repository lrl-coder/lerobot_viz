# LeRobot 数据集可视化部署

以下步骤适用于 Ubuntu/Linux。可视化不需要 GPU，但需要 FFmpeg（含 `libx264`）。

## 1. 克隆仓库

```bash
git clone https://github.com/lrl-coder/lerobot_viz.git
cd lerobot_viz
```

## 2. 安装环境

使用 [uv](https://docs.astral.sh/uv/) 创建 Python 3.12 虚拟环境：

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

安装 CPU 版 PyTorch 和可视化依赖：

```bash
uv pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.1 \
  torchvision==0.22.1

uv pip install \
  datasets==2.21.0 \
  flask==3.1.3 \
  av==15.1.0 \
  numpy==2.2.6 \
  pandas==2.3.3 \
  h5py==3.14.0 \
  pillow==12.3.0 \
  requests==2.34.2 \
  packaging==25.0 \
  jsonlines==4.0.0
```

## 3. 运行

将 `/path/to/session_name` 替换为本地数据集目录：

```bash
source .venv/bin/activate

PYTHONPATH="$PWD/lerobot-src-v0.3.3/src" /root/autodl-tmp/lerobot_viz/.venv/bin/python -m lerobot.scripts.visualize_dataset_html \
  --root /path/to/session_name \
  --host 0.0.0.0 \
  --port 9090
```

本地模式会自动取 `--root` 的最后一级目录名作为 repo id。例如
`--root /root/autodl-tmp/single_left_ft300s/session_20260805_091643` 对应的 repo id 为
`session_20260805_091643`，无需再传 `--repo-id`。

浏览器打开：

```text
http://服务器IP:9090/local/session_name/episode_0
```

如果数据集目录下存在 `video/depth/episode_XXXXXX.h5`，页面会自动读取其中形如
`observation.images.depth.<camera>` 的三维深度数组，并放在同名 RGB 相机旁边展示。例如：

```text
/path/to/dataset/video/depth/episode_000000.h5
```

深度文件位于其他目录时，添加：

```text
--depth-dir /path/to/depth
```

派生视频只缓存在运行程序的本地磁盘中，目录固定为数据根目录下的
`.lerobot-viz-cache/generated-videos/`。内嵌 RGB 图片、RGB 原始帧和深度视频分别位于
`embedded-images/`、`raw-images/` 和 `depth/`。`--output-dir` 仅控制 Web 服务临时文件位置，
不会改变视频缓存目录。

程序会在 Web 服务启动前遍历本次加载的全部 episode，将内嵌 RGB 图片、RGB 原始帧和深度
图像预先合成为 H.264 视频。第一次运行需要等待所有派生视频生成完毕；生成过程使用临时文件
并在编码完成后原子写入上述缓存目录。后续启动会保留并复用已有磁盘缓存，只补齐缺失或比
对应深度 HDF5 文件更旧的视频。网页请求只读取已经生成的缓存 MP4，不会触发即时编码。

数据源发生变化并需要强制重建所有派生视频时，添加：

```text
--rebuild-cache 1
```

所有 HTTP 响应仍携带 `Cache-Control: no-store` 等禁止缓存响应头，浏览器不会保存页面、静态
资源或视频缓存；持久缓存只存在于服务器磁盘，刷新页面无需使用 `Ctrl+Shift+R`。
深度颜色范围使用整段 episode 中有效像素的 1%–99% 分位数固定归一化，零值和无效值
显示为黑色，因此播放时不会因逐帧缩放而闪烁。

如只需预览部分 episode，可在运行命令中添加：

```text
--episodes 0 1
```

指定 `--episodes` 时只预生成并展示所选 episode；不指定时处理数据集中的全部 episode。
