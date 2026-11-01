# 结果说明

本算法对各阶段的计算时间均设有上限。因此，即使固定随机数种子，不同计算机的运算速度和系统负载仍可能影响限时内搜索到的候选方案，使重跑程序生成的最终调度方案存在一定差异。本附件代码已将每个核数配置的计算时间上限提高至600秒，高于生成附件result中结果时采用的时间预算。更充裕的搜索时间有助于探索更多候选方案，为进一步改善结果提供机会；实际结果仍可能因运行环境和搜索过程的差异而有所不同。附件result中记录了本程序在作者电脑跑出来的所有调度方案以及将方案代入官方核验程序得到的最终结果。

# 运行说明

使用 Python 3.10 或以上版本，无需安装第三方依赖。完整解压压缩包并保持目录结构；以下命令均在包含 `code`、`result` 和 `plot data` 的根目录执行。

## 生成调度方案

```bash
python code/problem1_code/main.py case_001 2
python code/problem2_code/main.py case_001 2
python code/problem3_code/main.py case_001 2
```

第一个参数为 `case_001` 至 `case_100`，第二个参数为核数 `1` 至 `5`。

输入图及固定配置位于 `code/100case_data/`。生成的方案保存至对应程序目录的 `output/<case>/m<核数>/<case>_multicore_res.json`，例如 `code/problem1_code/output/case_001/m2/case_001_multicore_res.json`。

每个核数配置最多计算 600 秒；多核计算可能先生成较低核数的候选方案，因此一次命令的总耗时可能超过 600 秒。限时搜索的结果可能随计算机性能和系统负载变化。

## 官方评测

评测时显式传入原始图、方案及配置路径。以下命令复核 `result` 中的已有方案；复核新生成方案时，将第二个路径替换为对应的 `code/problemN_code/output/…` 路径。

第一问示例：

```bash
python result/problem1_result/official_runtime/multicore_cut_evaluate_problem_1.py code/100case_data/case_001.json result/problem1_result/case_001/m2/case_001_multicore_res.json --config code/100case_data/config.txt -o evaluation_output/problem1/case_001/m2/result.json --trace-output evaluation_output/problem1/case_001/m2/trace.json --log-output evaluation_output/problem1/case_001/m2/log.txt
```

第二问示例：

```bash
python result/problem2_result/official_runtime/multicore_cut_evaluate_problem_2.py code/100case_data/case_001.json result/problem2_result/case_001/m2/case_001_multicore_res.json --config code/100case_data/config.txt -o evaluation_output/problem2/case_001/m2/result.json --trace-output evaluation_output/problem2/case_001/m2/trace.json --log-output evaluation_output/problem2/case_001/m2/log.txt
```

第三问无二级缓存示例（使用第二问评测器）：

```bash
python result/problem2_result/official_runtime/multicore_cut_evaluate_problem_2.py code/100case_data/case_064.json result/problem3_result/case_064/m5/no_l2/case_064_multicore_res.json --config code/100case_data/config.txt -o evaluation_output/problem3_no_l2/case_064/m5/result.json --trace-output evaluation_output/problem3_no_l2/case_064/m5/trace.json --log-output evaluation_output/problem3_no_l2/case_064/m5/log.txt
```

第三问只读缓存示例（使用第三问评测器）：

```bash
python result/problem3_result/official_runtime/multicore_cut_evaluate_problem_3.py code/100case_data/case_064.json result/problem3_result/case_064/m5/read_only_cache/case_064_multicore_res.json --config code/100case_data/config.txt -o evaluation_output/problem3_read_only_cache/case_064/m5/result.json --trace-output evaluation_output/problem3_read_only_cache/case_064/m5/trace.json --log-output evaluation_output/problem3_read_only_cache/case_064/m5/log.txt
```

结果分别写入 `evaluation_output` 下的完整结果文件 `result.json`、时间线文件 `trace.json` 和文本摘要 `log.txt`，输出目录自动创建。缓存峰值和详细执行时段可在完整结果文件中查看，时间线文件可用 Perfetto 打开。`config.txt` 中的赛题固定参数应保持原值。

# 算法来源与许可证

本项目由 Haoyang-Ying 编写的原创贡献采用 **Apache License 2.0**，版权署名为
`Copyright 2026 Haoyang-Ying`。完整许可证见根目录的 [LICENSE](../LICENSE)，
第三方来源、归属和适用范围见 [NOTICE](../NOTICE)。

BSPg 相关实现参考以下论文的算法 1 及作者公开的 `GreedyBSP` 代码：

> Pál András Papp, Georg Anegg, Aikaterini Karanasiou, and Albert-Jan N. Yzelman.
> *Efficient Multi-Processor Scheduling in Increasingly Realistic Models*.
> SPAA 2024, pp. 463–474. DOI: [10.1145/3626183.3659972](https://doi.org/10.1145/3626183.3659972).

- [论文全文](https://arxiv.org/abs/2404.15246)
- [作者配套代码](https://github.com/Algebraic-Programming/Artifacts/tree/master/SPAA_2024_Efficient_Multi-Processor_Scheduling)

上游代码采用 Apache-2.0，保留 `Copyright 2024 Huawei Technologies Co., Ltd.`
及作者署名。当前 Python 实现针对赛题任务模型进行了适配，并包含分支拆分、任务合并、
增量评分及场景相关扩展；不同文件的改动范围有所不同，不是上游原样发布版本。

`official_runtime` 目录中的赛题程序以及 `code/100case_data` 中的输入图和固定参数
保留原权利人的权利与适用条款。本包未附其独立再分发许可，公开这些材料前应核实
赛题原始发布条款；本项目的 Apache-2.0 声明不替代这部分授权。

本代码由 Codex 辅助完成。
