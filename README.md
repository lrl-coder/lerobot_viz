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

如只需预览部分 episode，可在运行命令中添加：

```text
--episodes 0 1
```
