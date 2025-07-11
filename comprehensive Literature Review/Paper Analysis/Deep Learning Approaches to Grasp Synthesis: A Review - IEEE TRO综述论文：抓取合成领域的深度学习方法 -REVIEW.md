# IEEE TRO综述论文：抓取合成领域的深度学习方法

**作者**: 王仕贤，杨少博  
**机构**: CAAI认知系统与信息处理专委会  
**日期**: 2024年11月05日 00:01  
**地点**: 北京

---

## **TRANSACTIONS ON ROBOTICS综述论文：抓取合成领域的深度学习方法**

抓取是机器人在现实世界中操纵物体的基本技能之一，涉及在一组接触点上施加力和扭矩来控制物体的运动。近些年，深度学习方法的突破性研究使机器人在抓取方面取得了快速进展。

近期，就职于澳大利亚克莱顿市莫纳什大学的 Rhys Newbury 在 **TRANSACTIONS ON ROBOTICS** 期刊上发表了综述论文 *“Deep
Learning Approaches to Grasp Synthesis: A Review”*，总结了近十年六自由度抓取合成的各类方法。

### 常见方法：

- **基于采样的方法**
- **直接回归方法**
- **强化学习方法**
- **范例方法**

此外，论文还总结了两种围绕抓取的 **支持方法**，它们主要使用深度学习来支持抓取过程、形状近似和可见性。

![img_47.png](../../1/assests/screenshot/screenshotBy12302024/img_47.png)

---

### **论文地址**:

[https://ieeexplore.ieee.org/abstract/document/10149823](https://ieeexplore.ieee.org/abstract/document/10149823)

---

随着计算机视觉中数据驱动方法的增加以及深度学习方法（特别是结合大规模数据时）使得机器人能够实现包括端到端的操作学习、双手协调抓取、手内灵巧操作、在杂乱环境中的拾放等操作任务。
![img_48.png](../../1/assests/screenshot/screenshotBy12302024/img_48.png)
这篇工作将主要介绍了深度学习在六自由度抓取合成上的常见方法、深度学习在抓取过程中的支持方法以及数据集设计方法。

---

## **深度学习在抓取中的应用方向**

### 1. 深度学习方法在六自由度抓取中的应用

常见的深度学习方法在六自由度抓取中主要包括：

- **基于采样的方法**
- **直接回归方法**
- **强化学习方法**
- **范例方法**

通过对比这些方法，论文提供了对当前深度学习在六自由度抓取合成领域研究进展的全面概述。每种方法各有优缺点，需根据具体应用场景、处理速度需求以及可用数据类型来选择合适的抓取策略。
![img_49.png](../../1/assests/screenshot/screenshotBy12302024/img_49.png)

---

### 2. 深度学习在抓取过程中的支持方法

论文提出了两种围绕抓取的 **支持方法**，它们主要利用深度学习来辅助抓取过程：

- **形状近似**
- **可见性识别**

这些支持方法不仅提高了抓取本身的成功率，还通过对物体形状的理解和可供性识别，使机器人能够更好地适应复杂的操纵任务。
![img_50.png](../../1/assests/screenshot/screenshotBy12302024/img_50.png)

---

### 3. 数据集设计

论文强调了数据集在机器人抓取研究中的重要性，包括：

- 选择适当的对象集
- 生成和使用程序生成的数据集
- 利用专家演示数据集
- 选择合适的数据表示和网络架构

这些因素对于训练有效的抓取模型和确保研究结果的可比性与可复现性至关重要。

---

### 4. 总结

论文对深度学习在六自由度抓取合成领域的研究现状进行了总结，并提出了未来研究的方向：

- **多样化环境下的六自由度抓取研究**
- **发布算法的可执行代码**
- **采用一致的性能指标**
- 强调 **多模态传感** 的重要性（如触觉、听觉等感知模态），扩展机器人抓取与复杂操作研究。

---

## **参考文献**

R. Newbury et al., *"Deep_Learning_Approaches_to_Grasp_Synthesis: A Review"*, in IEEE Transactions on Robotics, vol. 39,
no. 5, pp. 3994-4015, Oct. 2023, doi: [10.1109/TRO.2023.3280597](https://doi.org/10.1109/TRO.2023.3280597).