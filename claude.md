# 设计目标

* 支持多种 method（主方法 + 多个 baseline），每个 method 可以由多个子模块组成（encoder、decoder、head、loss 等）；method 实现应可组合与复用。
* 支持独立的 inference pipeline（predict、postprocess、解析 checkpoint 的细节输出与可视化）。
* 训练 → 多种后处理分析（not only evaluate once）：自动导出多种评估 artifact（confusion matrix、embedding 可视化、per-class metrics、saliency maps、生成样例等）。
* 向 LLM / diffusion 演化时，现有接口应最小改动（主要在 model 层替换与 trainer config）。
* 保留简洁性：不支持多 datamodule 组合（统一预处理输出格式），但提供 `preprocess/` 插件点。

# 最小目录结构（v1.0，带扩展点）

```
project_root/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Hydra 入口（run_train / run_optuna / run_infer）
│   ├── train.py                   # run_train / run_optuna / run_infer_pipeline
│   ├── entrypoints.py             # 小型适配器：CLI -> train.run / infer.run 等
│   ├── data/
│   │   ├── preprocess.py          # 全局 preprocess（统一输出格式）
│   │   └── wine_datamodule.py
│   ├── methods/                   # 所有方法与 baseline 的集合（必须结构化）
│   │   ├── __init__.py
│   │   ├── our_method/
│   │   │   ├── model.py           # pure nn.Module 组件（encoder/decoder/head）
│   │   │   ├── lit_module.py      # LightningModule：training/val/test/predict_step
│   │   │   └── config.yaml
│   │   ├── baseline_x/
│   │   └── registry.py            # helper：根据 name 返回 method factory
│   ├── inference/                 # inference pipeline + analysis utilities
│   │   ├── pipeline.py            # predict, postprocess, aggregate artifacts
│   │   └── analyzers.py           # checkpoint -> granular analysis
│   ├── callbacks/                 # custom callbacks (checkpoint, oom handler, analysis)
│   ├── utils/
│   │   ├── mlflow_utils.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── scripts/                   # small runnable scripts (e.g., export_ckpt.py)
│
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   ├── method/
│   │   ├── our_method.yaml
│   │   └── baseline_x.yaml
│   ├── trainer/
│   ├── inference/
│   ├── logger/
│   ├── mode/
│   ├── optuna/
│   └── hydra/launcher/slurm.yaml
│
├── tests/                         # unit tests: instantiation + one-batch run
└── README.md
```

# 核心设计约定（接口规范 — 必须严格遵守）

1. **Method 分层**

   * `methods/<name>/model.py`：只包含 `torch.nn.Module` 组件（可多个类：Encoder, Decoder, Head）。**不得**包含训练逻辑、hydra、logger。
   * `methods/<name>/lit_module.py`：继承 `pl.LightningModule`，实现 `training_step/validation_step/test_step/predict_step/configure_optimizers`。**predict_step** 必须输出标准字典：`{"pred": Tensor, "meta": {...}}`（便于后续 pipeline 聚合）。
   * 每个 method 在 `method/<name>/config.yaml` 中声明其超参与 `__target__` 路径，Hydra 通过 `instantiate(cfg.method)` 创建 `lit_module`（注意：instantiate 返回 LightningModule，Trainer.fit 接受）。

2. **数据规范**

   * `preprocess.py` 提供 `transform_raw_to_tensor(raw) -> (X_tensor, y_tensor)`。所有 datamodule 负责把原始数据统一转换为相同格式的 tensors。
   * Datamodule 必须实现 `setup(stage)` 并创建 `train_ds/val_ds/test_ds`，并返回 DataLoader。Batch shape 与 model.forward 输入应兼容。

3. **train.run 的 contract**

   * 函数签名：`def run_train(cfg: DictConfig) -> Dict[str, Any]`
   * 必须执行：seed → instantiate datamodule → instantiate model（由 cfg.method.*target*）→ instantiate logger → instantiate Trainer → `trainer.fit()` → `trainer.test()`（若 cfg.mode 要求）→ 返回 `metrics`（字典，包含 final val/test 指标）。

4. **Optuna**

   * `run_optuna(cfg)`：只修改 cfg 的副本（不可原地变更），`objective(trial)` 返回单个数值指标（例如 `val_loss`），外层 `study.optimize`。Optuna 与 MLflow 的集成通过 `MLflowCallback`，每个 trial 会写成 MLflow child run。Storage 建议使用 RDB（configurable）。

5. **Infer / Analysis pipeline**

   * `run_infer(cfg)` 用于：加载 checkpoint → `trainer.predict()`（或 `lit_module.predict_step`）→ 调用 `inference.pipeline.postprocess` → 调用 `analyzers` 生成 artifacts（metrics files, plots）→ 将 artifacts 上传到 MLflow（或写到共享目录）。
   * `analyzers` 模块包括：per-class metrics、calibration、confusion matrix、TSNE/UMAP embedding、saliency/attribution（接口可选）。

6. **Checkpoint 与 Artifact 管理**

   * Lightning checkpoint 的命名格式：`{method}-{dataset}-{timestamp}-epoch={EPOCH}-val={VAL:.4f}.ckpt`。
   * 必须在训练结束或异常保存一个 `latest` 指向文件（简单 symlink 或 JSON 记录）。
   * MLflow：在每个 run 中记录 `config.yaml`（完整合并的 cfg）以及关键 artifacts（best checkpoint, metrics json, sample outputs, plots）。`mlflow_utils.py` 提供统一 API：`log_artifact(path, tag)`。

7. **OOM 与 Robustness**

   * 在 `training_step` 周期捕获 `RuntimeError` 含 "out of memory" 的情形，执行 `torch.cuda.empty_cache()`、保存临时 checkpoint、记录失败原因到 artifact 并 `raise`（或使用自定义 exit code 让 submitit 重排）。
   * 提供 `callbacks/oom_handler.py`（可注入 Trainer callbacks）。

8. **扩展到 LLM / Diffusion**

   * 只要 `methods/<name>/model.py` 中的 `nn.Module` 依然返回同样类型的 forward/predict contract（tensors / dicts），Trainer/Datamodule/Inference pipeline 可以复用。LLM/diffusion 可能需要不同 `trainer.strategy`（e.g., deepspeed, fsdp）——这些只在 `configs/trainer/` 中配置，不改变代码层接口。

# 配置约定（Hydra）

* `configs/config.yaml` 顶层 defaults 指向 dataset、method、trainer、logger、mode、optuna、hydra/launcher。
* `method` config 必须包含 `_target_` 指向 `methods.<name>.lit_module.<Class>`（即 instantiate 后得到 LightningModule）。
* `inference` 配置包含 `checkpoint_path`、`postprocess` 选项与 `analyzers` 列表（按需开启）。

# 测试与验收准则（Acceptance）

为保证 template 正确性，必须通过下列自动/手动检验（每项通过为绿色）：

1. Instantiation checks（unit）

   * `instantiate(cfg.dataset)`、`instantiate(cfg.method)`、`instantiate(cfg.logger)` 成功。
2. Single-batch forward（unit）

   * 从 datamodule 取一 batch，`model(batch_x)`能前向。
3. Fast dev run（integration）

   * `python -m src.main mode=debug dataset=wine method=mlp`：`fast_dev_run=True`，trainer 完成。
4. Full run（integration）

   * `python -m src.main dataset=wine method=mlp`：训练若干 epoch，val_loss 降低。
5. Optuna 测试（integration）

   * `optuna.n_trials=3`，每个 trial 产生 MLflow run 与 metrics，study 能完成。
6. Inference & analyzers（integration）

   * `run_infer` 能加载 best ckpt，输出 artifacts（confusion matrix、one sample predictions）。
7. SLURM submit（smoke）

   * `hydra/launcher=slurm` 正确将 job 排队并运行（若能访问 SLURM）。

# 反模式与强制约束（复述）

* 模型不可直接读取 cfg（只能通过 `save_hyperparameters()` 保存必要超参）；如需 cfg，请在 lit_module 初始化时传入特定字段（明确列出）。
* 不允许在训练逻辑里写网络结构（模型实现只在 methods 下）。
* 不允许在 inference pipeline 修改训练流程或 cfg。

