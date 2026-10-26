# Bicycle mixed-degradation Fused30k 快速趋势实验

数据集：`/home/ubuntu/slocal/OneRestoreSynth3DGS_4_moderate/scenes/bicycle_low_haze_rain_snow`

运行设备：GPU0（RTX A6000）  
运行时长：约 14 分 15 秒（包括 1k、3k、5k 全测试集评测）  
训练视图 / 测试视图：169 / 25  
初始 COLMAP 点数：138,766  
3k 后高斯数：461,701

快速版把 30k 的三个启用点等比例提前，用于检查训练方向：

- 1–1.5k：基础 DCP+tone 3DGS；
- 1.501k–3.5k：基础 3DGS与物理解耦同时更新；
- 3.501k–5k：再加入独立、有界的 SH-DC Teacher residual；
- 基础高斯全程每步更新，增密在 3k 结束。

## 指标

| 输出 | Iteration | Clean PSNR ↑ | Clean SSIM ↑ | Clean LPIPS ↓ | Degraded PSNR ↑ | Degraded SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Fused pilot base | 1,000 | 18.8859 | 0.4236 | 0.5676 | 16.3309 | 0.4839 |
| Fused pilot base | 3,000 | 19.3308 | 0.4797 | 0.4958 | 21.0902 | 0.5431 |
| Fused pilot base | 5,000 | 19.6698 | 0.5172 | 0.4561 | 21.5827 | 0.5617 |
| Fused pilot corrected | 5,000 | **19.6765** | **0.5173** | 0.4563 | 21.5784 | **0.5618** |
| 历史 DCP+tone+3DGS pilot | 3,000 | 19.4050 | 0.4928 | 0.4756 | — | — |
| 历史 DCP+tone+3DGS | 30,000 | 20.0210 | 0.5824 | 0.3426 | — | — |
| 历史串行 60k best | 42,000 | 20.0299 | 0.5825 | 0.3427 | 21.4040 | — |

3k 对同 iteration 历史 pilot 的差值为 -0.0742 dB PSNR、-0.0131 SSIM、+0.0202 LPIPS。3k 到 5k 的基础输出继续提高 0.3390 dB PSNR、0.0375 SSIM，LPIPS 降低 0.0397，训练曲线仍在向历史 30k 终点靠近。

Teacher 在最后 1.5k 中将 PSNR 提高 0.0067 dB、SSIM 提高 0.00002，LPIPS 波动 +0.00019。其 RGB residual 绝对值均值为 0.000522，最大值为 0.03；1,385,103 个 residual 系数中有 42,282 个非零。当前权重体现为小幅修色，基础输出保持独立。

本次快速运行中的 residual 用于 Teacher 学习和 corrected 评测。运行结束后，正式入口又补上了 `corrected stop-gradient → physics` 的前向连接；用 5k checkpoint 做渲染接口检查时，单张图的平均绝对输入变化为 0.000680。该连接不改变基础分支的监督或梯度，正式 30k 将使用补齐后的实现。

## 产物

- `metrics/base/`：1k、3k、5k 基础指标；
- `metrics/corrected/`：5k Teacher 修正指标；
- `point_cloud/iteration_3000` 与 `iteration_5000`：高斯点云；
- `physics/`：3k 与 5k 物理分支状态；
- `residual/iteration_5000.pt`：独立 SH-DC residual；
- `chkpnt5000.pth`：5k checkpoint。
