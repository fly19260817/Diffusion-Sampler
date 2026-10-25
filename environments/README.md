# 本机环境清单

配置读取日期：2026-10-25。仅进行了系统/版本信息读取，未安装、更新或创建环境，未运行测试。

已读取的本机信息：

- Windows 10 专业版，10.0.19045，x86_64。
- NVIDIA RTX 4090，24564 MiB 显存，驱动 610.74。
- nvidia-smi 报告驱动支持 CUDA 13.3；CUDA_PATH 指向本机 Toolkit 12.6。
- Windows Conda 位于 E:\anaconda3，版本 23.1.0，base Python 3.9.13。
- 默认正在运行的发行版为 Ubuntu-22.04，WSL2；系统为 Ubuntu 22.04.4 LTS x86_64。
- WSL Conda 位于 /home/user/miniconda3，版本 25.3.1，base Python 3.13.2。
- WSL 的 nvidia-smi 能识别 RTX 4090 和驱动 610.74。
- 另有停止状态的 Ubuntu（WSL2）及 Ubuntu-20.04（WSL1）；WSL 清单针对 Ubuntu-22.04。

清单选择 Python 3.11，不修改任一 base 环境。两端采用官方 torch 2.9.1+cu126 /
torchvision 0.24.1+cu126 wheel；运行使用 wheel 自带 CUDA runtime，不需要清单再安装
系统 CUDA Toolkit。WSL 使用宿主机提供的 GPU 驱动，不在清单加入 Linux 显卡驱动。

FLUX.3 官方文档使用 main 版 ModularPipeline，故两套清单统一固定 Diffusers Git 提交：
aaf57d03a3edfbd9de457d2872417a92f149a91d。Transformers 固定为查询到的 5.20.2。
其余依赖采用版本范围；未经依赖解析或运行验证，不能当作完整 lockfile。
需要 Git 下载源码依赖，因此清单包括 git。没有加入 flash-attn/xformers 等编译型优化依赖。

创建环境及 editable 安装命令见项目 README。环境文件不含 Windows prefix，WSL 不使用 Windows Python。
本地权重路径在 Windows 为 H:\Diffusion-Sampler\sd35-medium-diffusers，
在 WSL 默认挂载规则下为 /mnt/h/Diffusion-Sampler/sd35-medium-diffusers。

24GB 显存下，较大 text encoder/transformer 的组合可能需要 CPU offload、量化及足够主内存。
经典 pipeline 提供 model CPU offload；FLUX.3 使用 leaf-level group offload，可能较慢。
先完成组件替换，再选择 to("cuda") 或 enable_cpu_offload()，不要同时叠加两种设备策略。
未对本机测量模型峰值内存或推理速度。

版本依据：
[PyTorch 官方 wheel 清单](https://pytorch.org/get-started/previous-versions/)、
[FLUX.3 官方说明](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux3)、
[固定 Diffusers 源码](https://github.com/huggingface/diffusers/tree/aaf57d03a3edfbd9de457d2872417a92f149a91d)。
