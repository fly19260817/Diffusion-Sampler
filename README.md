# Diffusion-Sampler

基于官方 Diffusers 的可组合采样接口，覆盖 SD 3.5、FLUX.1、FLUX.3 和 Qwen-Image 文生图。
模型编码、guidance、latent 打包及 VAE 解码交由各自的官方实现负责；本项目负责组件装配、
采样接口及可独立替换的数值求解器。


## 文件

- `src/diffusion_sampler/pipeline.py`：统一 pipeline、组件加载及替换、结果对象。
- `src/diffusion_sampler/scheduler.py`：`FlowScheduler`，保留模型时间网格配置并接入 solver。
- `src/diffusion_sampler/solver.py`：Euler / AB2 与模型输出参数化。
- `environments/environment.windows.yml`、`environments/environment.wsl.yml`：各平台 Conda 清单。
- `environments/README.md`：本机配置、环境创建说明和限制。
- `examples/generate.py`：命令行采样示例。

## 环境准备

以下命令仅供之后手动使用，本次没有执行：

Windows PowerShell：

```powershell
Set-Location H:\Diffusion-Sampler
conda env create -f environments/environment.windows.yml
conda activate diffusion-sampler-windows
python -m pip install -e . --no-deps
```

WSL2 Ubuntu 22.04：

```bash
cd /mnt/h/Diffusion-Sampler
source /home/user/miniconda3/etc/profile.d/conda.sh
conda env create -f environments/environment.wsl.yml
conda activate diffusion-sampler-wsl
python -m pip install -e . --no-deps
```

## 使用模型原始采样设置

```python
import torch
from diffusion_sampler import SamplingPipeline

pipe = SamplingPipeline.from_pretrained(
    r"H:\Diffusion-Sampler\sd35-medium-diffusers",
    family="sd35",  # 也支持 auto，从 model index 识别
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.enable_cpu_offload()
result = pipe("A small observatory under a starry sky", seed=42)
result.images[0].save("sample.png")
```

不传 scheduler / solver 时使用加载到的原始 scheduler 对象及其数值方法。
steps、guidance、尺寸等未指定参数不由本项目覆盖，沿用官方 pipeline/config 默认值；
这些默认值不一定等同于每个模型卡推荐的配方，例如 FLUX.1-schnell 可以显式传
`num_inference_steps=4, guidance_scale=0.0`。

各模型使用：

- SD 3.5：`family="sd35"`，本机已有 `sd35-medium-diffusers`。
- FLUX.1：`family="flux1"`，例如 `black-forest-labs/FLUX.1-dev`。
- FLUX.3：`family="flux3"`，例如 `black-forest-labs/flux-3-image-klein-base`。
  采用官方 ModularPipeline，单独加载组件，BF16 加载时默认保留 VAE 为 float32。
  蒸馏版也保留其原生流程；自定义 solver 限定为 flow Euler，因为其输出对应特定目标时间步。
- Qwen-Image：`family="qwen-image"`，例如 `Qwen/Qwen-Image`。
  官方参数（例如 `true_cfg_scale`、negative_prompt）直接透传。
  此适配器面向 Qwen-Image 文生图，其他编辑版或后续架构可通过 `from_backend()` 包装。

这里给出的远程模型名不会被自动下载；只有用户之后实际调用加载接口时才访问模型。
`local_files_only=True` 也会传给 FLUX.3 的剩余组件加载。
显式 ComponentSource 的加载选项独立管理，需要离线时也要为它设置 local_files_only。

## 模块拼接

加载时传入对象，或者描述一个需要单独加载的组件：

```python
from diffusers import AutoencoderKL, SD3Transformer2DModel
from diffusion_sampler import ComponentSource, SamplingPipeline

root = r"H:\Diffusion-Sampler\sd35-medium-diffusers"
pipe = SamplingPipeline.from_pretrained(
    root, family="sd35", local_files_only=True,
    components={
        "vae": ComponentSource(
            AutoencoderKL, root,
            {"subfolder": "vae", "local_files_only": True},
        ),
        "transformer": ComponentSource(
            SD3Transformer2DModel, root,
            {"subfolder": "transformer", "local_files_only": True},
        ),
        # "text_encoder": 已加载的 encoder,
        # "tokenizer": 与该 encoder 匹配的 tokenizer,
    },
)
```

`components` 还支持 text_encoder_2/3、tokenizer_2/3 以及对应模型声明的其他组件；
拼错名称会报错。省略的组件来自原模型，显式 None 则禁用该组件（调用时需补足对应 embeddings）。
组件必须具有相容的结构、通道数、tokenizer 和训练约定，不能任意跨架构混搭。

后续可调用 `pipe.replace_components(vae=new_vae, text_encoder=new_encoder)`；
请先完成替换，再配置设备及 offload，以免旧 hooks 残留。

也可以完全从现有组件构造：

```python
pipe = SamplingPipeline.from_components(
    family="sd35", scheduler=my_scheduler,
    transformer=my_transformer, vae=my_vae,
    text_encoder=clip_l, text_encoder_2=clip_g, text_encoder_3=t5,
    tokenizer=tokenizer_l, tokenizer_2=tokenizer_g, tokenizer_3=tokenizer_t5,
)
```

FLUX.3 的 from_components 默认使用 Flux3AutoBlocks，也允许传入 `blocks=my_blocks`
（例如蒸馏版本的官方 blocks）。需提供所选 blocks 所需的全部组件。
`SamplingPipeline.from_backend(native_pipeline)` 可以包装已构建的经典或模块化 pipeline；
`pipe.backend` 保留 LoRA、callbacks、模块化 output 选择等官方扩展入口。

## Scheduler 与 solver 分别替换

```python
from diffusion_sampler import FlowScheduler, Solver

# 只替换 solver：复制原始 scheduler 配置，保留 SD3 shift 或 FLUX/Qwen 动态 shift。
pipe.configure_sampling(solver=Solver(method="ab2"))

# 只替换 scheduler：使用该 scheduler 自带的求解方式。
schedule = FlowScheduler.from_config(pipe.scheduler.config, shift=4.0)
pipe.configure_sampling(scheduler=schedule)

# 同时替换时间网格配置和数值求解器。
pipe.configure_sampling(scheduler=schedule, solver=Solver(method="euler"))

# 恢复本次加载时记录的原始模型 scheduler。
pipe.configure_sampling()
```

FlowScheduler 沿用官方静态/动态 shift、mu、Karras/exponential/beta 网格配置，
以及自定义 sigmas/timesteps 接口。自定义 sigma 是上游 set_timesteps 的输入网格，
仍会接受其 shift 等变换；不要重复手动 shift，不要把末端零附加进去（上游会添加终点）。
动态 shift 模型通常由官方 pipeline 根据分辨率计算 mu；独立使用 scheduler 时需自行提供。

任意官方 scheduler 可通过 scheduler 参数注入，但需与模型及官方采样循环兼容。
独立 solver 目前仅桥接 FlowMatchEulerDiscreteScheduler / FlowScheduler；
不会把 DDPM/VP 或 Heun 的重复时间网格静默当作 rectified flow。

SolverProtocol 是普通接口基类，可选择继承；不做运行时协议检查。自定义 solver 实现 `reset()` 与
`step(model_output, sample, sigma, sigma_next)` 即可，每步一次模型评估。
AB2 使用变步长系数，第一步用 Euler 启动，每次 set_timesteps 重置历史。
接口不支持需要额外模型评估的 Heun/RK 求解器。自定义 solver 模式下不支持
stochastic_sampling、invert_sigmas、per-token timesteps 或 churn，会明确报错。

## 输出参数化

Solver 对应固定的线性 rectified-flow 路径：

```text
x(s) = (1-s) * x0 + s * epsilon
dx/ds = epsilon - x0
```

支持：

- `flow_prediction`：模型直接输出 epsilon - x0，四类官方适配器均使用此设置。
- `epsilon`：模型输出噪声，导数为 (epsilon - x)/(1-s)。
- `sample`：模型输出 x0，导数为 (x - x0)/s。
- `v_prediction`：定义为 (1-s)*epsilon - s*x0；
  导数为 ((2s-1)*x + v)/((1-s)^2+s^2)。

epsilon 在 s=1 不可逆，x0 在 s=0 不可逆；会报错而非通过加小量伪造结果。
v 的定义使用上述线性系数，不等同于 VP/DDPM 的归一化噪声轨迹。
改 prediction_type 只是在描述模型真实输出，不能把现成 flow 模型变为噪声预测模型；
统一 pipeline 因此拒绝给四类 flow 模型直接配置非 flow 的内置 Solver。

对于自己实现的、确实输出 epsilon/x0/v 的 rectified-flow 模型，可以直接组合：

```python
from diffusion_sampler import FlowScheduler, Solver

scheduler = FlowScheduler(shift=1.0).set_solver(Solver("euler", prediction_type="epsilon"))
scheduler.set_timesteps(sigmas=[0.99, 0.75, 0.5, 0.25])
# x 是由调用方初始化在 sigma=0.99 的 latent；model_epsilon 输出真实的 epsilon。
for t in scheduler.timesteps:
    prediction = model_epsilon(x, t)
    x = scheduler.step(prediction, t, x).prev_sample
```

只在 s<1 开始需要与该 sigma 对应的初始样本；不能假设纯高斯就是精确的初态。
求解器以 float32 计算更新（float64 输入保留双精度），再恢复 latent dtype。
实例包含历史状态，不应在并发请求之间共享。scheduler 保存仅保存上游网格配置，
不保存运行时 solver 或历史；重新加载后需重新 set_solver。

## 命令行示例

```powershell
python examples/generate.py --model H:\Diffusion-Sampler\sd35-medium-diffusers --family sd35 --local-files-only --prompt "A quiet mountain lake" --cpu-offload --output outputs/sd35.png
```

添加 `--solver ab2` 使用自定义求解器；不传则使用模型原始方法。
添加 `--steps 28` 显式指定步数。WSL 将模型路径换为 `/mnt/h/Diffusion-Sampler/sd35-medium-diffusers`。

## 参考

- [SD3 官方 pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- [FLUX.1 官方 pipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [FLUX.3 官方 ModularPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux3)
- [Qwen-Image 官方 pipeline](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage)
- [官方 FlowMatchEulerDiscreteScheduler 源码](https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
