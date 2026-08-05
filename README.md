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

将 `/path/to/dataset` 替换为本地数据集目录，将 `dataset_name` 替换为数据集名称：

```bash
source .venv/bin/activate

PYTHONPATH="$PWD/lerobot-src-v0.3.3/src" python -m lerobot.scripts.visualize_dataset_html \
  --repo-id local/dataset_name \
  --root /path/to/dataset \
  --host 0.0.0.0 \
  --port 9090
```

浏览器打开：

```text
http://服务器IP:9090/local/dataset_name/episode_0
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

每次通过命令行启动程序时都会先清空派生视频缓存，因此数据变化后不会沿用上一次运行的
旧视频。同一次运行中，每路视频在第一次 HTTP 请求时生成，后续请求复用本地磁盘文件，
不重复编码。所有 HTTP 响应均携带 `Cache-Control: no-store` 等禁止缓存响应头，浏览器不会
保存页面、静态资源或视频缓存，刷新页面也无需使用 `Ctrl+Shift+R`。
深度颜色范围使用整段 episode 中有效像素的 1%–99% 分位数固定归一化，零值和无效值
显示为黑色，因此播放时不会因逐帧缩放而闪烁。

如只需预览部分 episode，可在运行命令中添加：

```text
--episodes 0 1
```
