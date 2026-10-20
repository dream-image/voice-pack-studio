# 本地语音包工作台

使用 Qwen3-TTS 和 MLX 在 Apple Silicon Mac 上生成中文语音，提供预设音色、设计音色和参考音频克隆三种模式。

## 启动

当前机器已安装依赖。在终端执行：

```sh
cd /Users/thousanddream/code/voice-pack-studio
uv run --frozen python app.py
```

保持该终端运行，然后用 Edge 打开 <http://127.0.0.1:7860/>。也可以在另一个终端执行：

```sh
open -a "Microsoft Edge" http://127.0.0.1:7860/
```

桌面的 `code` 是指向 `/Users/thousanddream/code` 的访达替身，启动时使用上面的真实路径。服务仅监听本机地址，不会自动打开浏览器。

## 停止

在运行服务的终端按 `Control + C`。关闭浏览器页面不会停止服务。

可用下面的命令检查端口；没有输出表示没有进程监听 7860 端口：

```sh
lsof -nP -iTCP:7860 -sTCP:LISTEN
```

## 首次安装或重建环境

需要 Apple Silicon Mac、`uv` 和 Python 3.12。进入项目目录后执行：

```sh
uv sync --frozen
uv run --frozen python app.py
```

Python 版本由 `.python-version` 指定，依赖版本由 `uv.lock` 锁定。项目的 Python 包下载源已配置为清华镜像。

## 文件位置

- 生成的 WAV、台词清单和 ZIP 素材包保存在项目的 `outputs/` 目录。
- 模型缓存保存在 `.cache/huggingface/`。首次使用某种模式需要联网下载模型，保留缓存可以避免重复下载。
- 人物和封面图片保存在 `assets/`。

导出的 ZIP 是音频素材合集，高德端需另行录入。
