# llama.cpp

> [!NOTE]
> **This is a personal fork: the RX 580 unshittification project.** Branch `ot-host-buffers` is upstream `master`
> plus a few patches, made to squeeze large MoE models out of an old GPU: **RX 580 8 GB** (Polaris/GCN4, Vulkan/RADV,
> PCIe 3.0, no fp16 arithmetic, no integer dot product, no cooperative matrices) + i7-10700 + 64 GB DDR4.
> Main target: Qwen3.8-Flash-Next (Q2_0 experts kept in RAM, dense part on the GPU, MTP speculative decoding).
>
> | Qwen3.8-Flash-Next on this box (llama-server) | stock config | now |
> |---|---|---|
> | generation | 12.8 tok/s | **18.2 tok/s** (+42 %) |
> | long-prompt processing | 62 tok/s | **133 tok/s** |
> | context that fits | 8K | 32K |
>
> **Patches** (each one is a separate commit on the branch):
>
> 1. **`-ot` / `--override-tensor` accepts pinned host buffer types** (`Vulkan_Host`, `CUDA_Host`, ...), so you can
>    pin only part of the tensors kept in host memory. On amdgpu, pinned (GTT) memory is capped at half of the RAM
>    by default (31 GiB here): pinning all the experts fails or loses the Vulkan device, not pinning them makes prompt
>    processing slower. With the patch, 44 of 48 expert layers of Qwen3.8-Flash-Next fit in the cap:
>    `-lm none --lazy-mode on --no-host -ot "blk\.([0-9]|[1-3][0-9]|4[0-3])\.ffn_.*_exps=Vulkan_Host,ffn_.*_exps=CPU"`
>    (long-prompt processing 104 -> 125 tok/s).
> 2. **Vulkan lightning indexer without shaderFloat16:** an fp32 variant, so GPUs without fp16 arithmetic (GCN/Polaris)
>    run it on the GPU instead of falling back to the CPU (122 -> 98 graph splits per token, decode +4-5 %).
> 3. **`llama-quantize ... COPY --tensor-type`** requantizes only the named tensors and copies the rest, e.g. to turn
>    the BF16 hyper-connection weights of a mixed-precision quant into Q8_0 (-0.55 GB of VRAM, same perplexity).
> 4. **Vulkan `multi_add` fusion on AMD drivers without RTE fp16** (GCN/Polaris): the 9 adds that combine the 10 routed
>    experts per layer run as fused dispatches (2980 -> 2591 dispatches per token, decode +2.5 %).
> 5. **Faster Q2_0 x Q8_0 `mul_mat_id` on the CPU (AVX2)** for MoE experts kept in RAM: the activations are rearranged
>    once per column to match the Q2_0 bit layout, and the weights are prefetched ahead. 8 threads: 33 -> 40 GB/s of
>    weights (the RAM limit of an i7-10700); Qwen3.8-Flash-Next decode +12.5 %.
>
> **Things learned on Polaris that are not patches** (useful if you run MoE models on an old AMD card):
>
> - **Pin the VRAM clock.** With experts on the CPU the GPU works in bursts (it waits for the CPU at every layer) and
>   amdgpu drops the memory clock to half (2000 -> 1000 MHz here) in the pauses. Fixing it
>   (`echo manual > power_dpm_force_performance_level; echo 2 > pp_dpm_mclk`) gives +2-3 % with MTP.
> - **Submit GPU work more often.** With ~53 nodes per GPU split, Vulkan only submits at 80 % of the split, so the GPU
>   waits for the CPU to record commands. `GGML_VK_MAX_NODES_PER_SUBMIT=16`: +1.4 %.
> - **`--spec-draft-sampling probabilistic --spec-draft-p-min 0`** when sampling with temperature: the draft samples
>   and the target verifies by rejection (exact), acceptance 72 -> 75 % at temp 0.8, +2.3 %.
> - **MTP helps a MoE with CPU experts** (+30 % here, 2 draft tokens is the sweet spot; 3-4 make the CPU read the
>   union of more experts, and that is already RAM-bound).
> - **`parallel = 1`** in llama-server for one user: the default 4 slots halve the speed and eat pinned memory.
> - **The Vulkan perf logger (`GGML_VK_PERF_LOGGER`) serializes every op**, so it overstates small ops (6-8 ms per
>   token in the logger, ~1.7 ms for real). To measure what removing an op is really worth, dispatch it twice;
>   skipping ops instead gives garbage activations and the MoE router collapses (fake speedups).
> - **An op that gets faster in `test-backend-ops` is not a faster model**: twice a matmul improved 25-60 % in
>   isolation and end-to-end throughput did not move. Always close with an end-to-end A/B (temperature 0, alternating).
> - On Polaris, sub-4-bit mat-vec shaders are ALU-bound (~330-430 G weights/s whatever the bits), so requantizing
>   Q3_K to a "faster" type does not pay off: what you win per byte you lose in bytes.
>
> The branch is rebased on upstream regularly. Everything else is unmodified upstream llama.cpp:
> please report issues to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), not here.

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[v1.0.0](https://huggingface.co/blog/ggml-org/llama-cpp-v1) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Description

The main goal of `llama.cpp` is to enable LLM (and MLLM) inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud.

The `llama.cpp` project is built on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Llama Chat, the built-in chat app, against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Llama Chat, the built-in chat app, against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [Text-to-speech (TTS) models](docs/multimodal/tts.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
