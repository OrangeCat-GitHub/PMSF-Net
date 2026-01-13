# **PMSF-Net-for-Event-Log-Repair**

本项目使用缺失感知卷积与多频谱融合（PMSF-Net）来修复事件日志中缺失的活动。

由于当初编写代码主要是为了研究和实验验证，所以可能存在一些冗余文件或不够完善的地方。我们上传的是精简和整理后的核心版本。

### **环境配置 (Environment Setup)**

```bash
# 建议使用 conda 创建环境
conda create -n pmsfnet python=3.8
conda activate pmsfnet

# 安装依赖
pip install torch numpy pandas scikit-learn
```
