# v27 数据布局

仓库不分发原始 EEG、问卷或个人信息。复现训练只需要三套已经预处理的 MAT 文件；标签、固定五折划分和 city 分段信息已脱敏后放在 `reproducibility/protocols/`。

配置中的三个数据目录必须具有以下布局：

```text
monifeixing_data_root/
  sub1_rest1_q.mat
  sub1_rest2_q.mat
  ...
  sub18_rest1_q.mat
  sub18_rest2_q.mat

vrq_data_root/
  sub1_rest01.mat
  sub1_rest02.mat
  sub1_task01.mat
  ...

city_data_root/
  Acquisition 01.mat
  Acquisition 02.mat
  ...
  Acquisition 26.mat
```

city 协议排除 `Acquisition 22.mat`，因此实际只要求其余 25 个文件。

具体文件集合不是按通配符猜测：`preflight` 会读取仓库内 manifest，逐个核验文件名和 SHA-256。缺文件、文件内容不同、五折身份重叠、checkpoint 不匹配都会立即报错。

MAT 数组要求：

- monifeixing 与 VRQ：键名 `data256`，至少 32 通道；使用通道 0–21、24–31；
- city：键名 `data256`，严格 37 通道；
- 采样率 256 Hz，窗口长度 1280 点（5 秒），不重叠切窗；
- 必须使用与 manifest 哈希一致的预处理版本，不能只把相似数据重命名后使用。

问卷工作簿不再是运行依赖。它们只用于生成当前仓库中的脱敏标签和协议；原始工作簿 SHA-256 保留在 `reproducibility/protocols/manifest.json` 中用于来源审计。

推荐配置：

```json
{
  "output_root": "outputs/reproduction_v27_seed2001",
  "paths": {
    "monifeixing_data_root": "/data/monifeixing",
    "vrq_data_root": "/data/vrq",
    "city_data_root": "/data/city",
    "pretrain_checkpoint": "local_assets/pretrained_femba_v27.ckpt"
  }
}
```

Windows 路径可直接写成 `D:/...`；在 WSL2 中会自动转换为 `/mnt/d/...`。
