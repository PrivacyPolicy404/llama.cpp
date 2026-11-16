# llama.cpp

> [!NOTE]
> **This is a personal fork: the RX 580 unshittification project.** Branch `ot-host-buffers` is upstream `master`
> plus a few patches, made to squeeze large MoE models out of an old GPU: **RX 580 8 GB** (Polaris/GCN4, Vulkan/RADV,
> PCIe 3.0, no fp16 arithmetic, no integer dot product, no cooperative matrices) + i7-10700 + 64 GB DDR4.
> Main target: [Qwen3.8-Flash-Next GSQ-RCO abliterated](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF)
> (125B MoE, ~6B active per token), a model about eight times larger than the card's memory.
>
> | The model as it runs here | |
> |---|---|
> | Main file | Q2_0 quant, 66 GB in two shards, read from NVMe |
> | Experts | Q2_0, 31.6 GiB, kept in RAM and computed on the CPU; 44 of 48 layers pinned (patch 1) |
> | Dense part | on the GPU (q3_K, q5_K, iq4_xs, q8_0), 4.7 GiB |
> | Hyper-connections | BF16 requantized to Q8_0 (patch 3): -0.55 GB of VRAM, same perplexity |
> | Speculative decoding | the repo's MTP draft block in Q4_0, 2 draft tokens, output trimmed to the 65536 most frequent tokens (patch 8) |
> | Context | 64K, K/V cache in Q8_0, one slot |
>
> | llama-server on this box | stock config | now |
> |---|---|---|
> | generation, short prompt | 12.8 tok/s | **20.0 tok/s** (+56 %) |
> | generation at 12K of context | did not fit | **19.8 tok/s** |
> | generation at 30K of context | did not fit | **18.4 tok/s** |
> | long-prompt processing | 62 tok/s | **134 tok/s** |
> | context that fits | 8K | **64K** (desktop off) |
> | VRAM in use | | 7.2 of 8 GiB |
>
> Generation at 12K / 30K is at temperature 0 over a saved context; the short-prompt figure is one sampled run of the
> panel's smoke test.
>
> **Setup, every option explained, what each patch changes and how to reproduce the numbers:
> [docs/rx580.md](docs/rx580.md).**
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
> 6. **Vulkan sparse flash attention from 4x on GCN with Q8_0 K/V.** Models that attend to a selected subset of the
>    cache (~2048 cells here) only took the sparse path when the cache was 16x the selection, a break-even measured
>    on RDNA3/RDNA4. On Polaris dense is slower and sparse wins from 4x (3 queries over 16K cells: 1155 -> 439 us),
>    so it now kicks in at ~8K of context instead of ~32K: with MTP, 17.1 -> 18.3 tok/s at 20K and 16.4 -> 17.4 at 30K.
> 7. **`GGML_VK_FIRST_SUBMIT_NODES=N`**: submit the first batch of each graph after N nodes, then every
>    `GGML_VK_MAX_NODES_PER_SUBMIT` as usual. With ~49 short GPU segments per token the GPU was idle while the first
>    16 nodes of each one were recorded. With N=2: decode +2.9 % without MTP, +0.8-1.4 % with MTP, identical output
>    (1, 4, 8 and doubling ramps are all worse).
> 8. **Reduced output vocabulary for the MTP draft (`d2t`).** If the draft file carries a `d2t` tensor with K token
>    ids, `output.weight` has only those K rows and the logits are scattered to the full vocabulary (same convention
>    as eagle3 / dflash; the idea is FR-Spec's frequency-ranked drafting). The target verifies against the full
>    vocabulary, so the output distribution is unchanged. With the 65536 most frequent of 248320 tokens: decode +2 %,
>    same accepted drafts, -0.24 GiB of VRAM and a draft file 263 MB smaller.
> 9. **q3_K / q5_K mat-vec shaders: scales folded into the weights once per row.** With 4 B columns those shaders
>    needed more than 128 VGPRs (147 / 143), so GCN ran one wave per SIMD instead of two: q3_K 10240x2560 took 226 us
>    with 3 columns and 369 with 4. Now 4 columns fit (128 / 115 VGPRs): 369 -> 265 us and 290 -> 221 us, and 1-3
>    columns are up to 7 % faster. Decode with 2 MTP drafts +1.1 % at 12K, +0.3 % at 30K. Lesson: check
>    `RADV_DEBUG=shaderstats` when a shader variant costs more than its work explains.
> 10. **iq4_xs mat-vec shader: the sub-block in two halves with 2+ columns.** Same threshold as patch 9, and here
>    it hit the everyday case: with 3 B columns (the verify batch of 2 MTP drafts) the shader needed 142 VGPRs.
>    Now 117: 10240x2560 with 3 columns 283 -> 173 us. Decode with 2 MTP drafts +0.7 % at 12K, +2.4 % at 30K.
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
> - **Quantize the KV cache on hybrid models.** Qwen3.8-Flash-Next has full attention in only 1 of 4 layers, so the
>   KV cache is small and `-ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0` takes 64K of context from 1.9 to 1.0 GiB with
>   KLD 0.0035 (98 % same top token), for ~3 % of speed.
> - **Size the MTP draft's batch separately: `--spec-draft-n-ubatch 512`.** By default the draft context reserves compute
>   buffers for the full `-ub` (867 MiB at 2048 here); with 512, 0.42 GiB less VRAM at the same speed.
> - **Leave ~0.3-0.4 GB of VRAM free and read per-process memory.** When llama.cpp asks for more VRAM than is free, the
>   kernel silently places the rest in GTT, so "VRAM used" looks fine (check `drm-memory-vram`/`-gtt` in
>   `/proc/PID/fdinfo`). And with the card at ~7.9 of 8 GB, long prompts (30K) run ~40 % slower (109 -> 64-68 tok/s);
>   short prompts don't show it.
>
> The branch is rebased on upstream regularly. Everything else is unmodified upstream llama.cpp:
> please report issues to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), not here.

![llama.cpp](https://raw.githubusercontent.com/ggml-org/llama-brand/refs/heads/master/logo/svg/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v1)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/pulls?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Asanmai%20OR%20author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Athelittlefireman%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3AmasterFoad%20OR%20author%3Aallozaur%20OR%20author%3AAndreas-Krebbel%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Achaxu01%20OR%20author%3Alhez%20OR%20author%3AAlekseiNikiforovIBM%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming%20OR%20author%3Aynankani)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

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
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Huawei Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
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
