# Cup Detection V1 vs V2 Failure Analysis

## 1. Experiment Goal

验证基于 V1 模型 Failure Analysis 进行针对性补数据后，是否能够改善模型在真实场景下的鲁棒性。

实验原则：

- V1 和 V2 使用相同模型：YOLOv8n
- 均使用 pretrained weights
- epochs = 30
- imgsz = 416
- batch = 4
- Val / Test 保持不变
- Challenge Set 保持不变
- 核心变量仅为 Training Data

---

## 2. Dataset Version

### Dataset V1

Train: 96 images  
Val: 12 images  
Test: 13 images  

Train 中无专门 Negative Samples。

### Dataset V2

在 V1 Train 基础上新增 32 张 targeted samples：

- 8 Pure Negative
- 8 Hard Negative
- 8 Small / Distant Positive
- 8 Hard Positive

最终：

Train: 128 images  
Val: 12 images  
Test: 13 images  

其中新增 16 张 Negative / Hard Negative 的 YOLO label 为空。

---

## 3. Why V2 Was Created

V1 在同分布 Test Set 上表现较高：

Precision: 0.995  
Recall: 1.000  
mAP50: 0.995  
mAP50-95: 0.896  

但在真实手机拍摄的 Challenge Set 上出现明显 Failure：

1. 将灯具、药瓶、水桶、插座等非 cup 物体误识别为 cup
2. 远距离、小尺度 cup 出现漏检
3. 新环境、复杂背景下泛化能力下降

因此将主要 Data Gap 判断为：

- Negative Samples 不足
- Hard Negative 不足
- Small / Distant Cup 不足
- Real-world Hard Positive 多样性不足

---

## 4. V1 vs V2 Standard Evaluation

| Metric | V1 | V2 |
| --- | ---: | ---: |
| Train Images | 96 | 128 |
| Val Precision | 0.998 | 0.998 |
| Val Recall | 1.000 | 1.000 |
| Val mAP50 | 0.995 | 0.995 |
| Val mAP50-95 | 0.886 | 0.875 |
| Test Precision | 0.995 | 0.996 |
| Test Recall | 1.000 | 1.000 |
| Test mAP50 | 0.995 | 0.995 |
| Test mAP50-95 | 0.896 | 0.910 |

结论：

V2 没有明显破坏原有同分布测试性能，Test mAP50-95 略有提升。

由于 Val / Test 数据规模较小，这些变化不应过度解读。

---

## 5. Challenge Set Evaluation

Challenge Set:

24 real-world smartphone images.

包含：

- cup positive samples
- negative samples
- hard negative objects
- small / distant cups
- occlusion
- complex backgrounds
- domain shift

图片级人工粗评估：

| Metric | V1 | V2 |
| --- | ---: | ---: |
| Positive-image Detection Rate | ~90% | ~60% |
| Negative-image False Positive Rate | ~42.9% | ~14.3% |
| Total Predicted Boxes | 39 | 9 |

注意：

以上为图片级 Failure Analysis 指标，不是正式 Object Detection mAP。

---

## 6. Failure Analysis

### Improvement

V2 显著降低了 False Positive。

典型改善包括：

- 药瓶误识别减少
- 插座误识别减少
- 保温瓶误识别减少
- 普通 Negative 场景误报减少
- 多余预测框数量明显下降

说明：

Negative Samples 和 Hard Negative Mining 对抑制误报有效。

### New Problem

V2 同时出现更多 False Negative。

典型情况包括：

- 远距离 cup 仍然漏检
- 部分裁切 cup 漏检
- 浴室等新环境中的 cup 漏检
- 部分复杂背景 cup 从 V1 的重复误报变成 V2 的完全不报

说明模型变得更加保守。

---

## 7. Root Cause

V1 数据以 Positive 为主：

Positive-heavy Dataset
 模型倾向于寻找 cup
 False Positive 较多

V2 加入大量有效 Negative / Hard Negative：

Negative supervision 增强
 False Positive 明显下降

但 V2 的 Hard Positive 仍然存在以下不足：

- 杯子类型较单一
- 多为同一个绿色马克杯
- 场景多样性有限
- 不同环境覆盖不足
- 不同外观 / 材质 / 颜色覆盖不足

因此：

False Positive 
但 False Negative 

形成明显的 Precision / Recall trade-off。

---

## 8. Next Data Strategy

下一轮不应继续大量增加 Negative。

Dataset V3 应优先补充 Diverse Hard Positive：

- 不同颜色 cup
- 不同材质 cup
- 不同形状 cup
- kitchen / bathroom / office 等不同场景
- small cup
- distant cup
- partial crop
- occlusion
- low light
- complex background
- cup containing other objects

目标：

保持 V2 较低的 False Positive，同时恢复困难场景下的 Recall。

---

## 9. Key Conclusion

Targeted Negative and Hard Negative supplementation significantly reduced false positives, but also exposed insufficient diversity in hard positive samples and caused recall degradation.

The next iteration should shift from adding more negative samples to collecting diverse hard positives across scenes, scales and object appearances.

---

## 10. Data Flywheel

Baseline V1  
 Standard Evaluation  
 Challenge Set  
 Failure Analysis  
 Targeted Data Collection  
 Dataset V2  
 Retraining  
 Same Challenge Set Re-evaluation  
 FP decreased but FN increased  
 New Data Gap identified  
 Define Dataset V3 strategy

