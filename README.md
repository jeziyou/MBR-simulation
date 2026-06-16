# 🌊 MBR工业仿真系统 v2.0

工程级膜生物反应器(MBR)仿真与优化平台 - Streamlit版本

## 功能特性

- **工程级计算引擎**：专业的能耗、剪切力、膜污染预测
- **3D交互可视化**：膜组件结构、气泡动画、污泥分布
- **性能仪表盘**：运行评分、优化建议、经济性分析
- **场景对比**：节能/均衡/高冲刷三模式一键对比
- **数据导出**：JSON/CSV/文本报告三种格式

## 在线演示

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_CLOUD_URL)

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. Fork 本仓库到您的 GitHub 账号
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 账号并选择本仓库
4. 主文件路径填写 `app.py`
5. 点击 Deploy

## 技术栈

- Python 3.10+
- Streamlit 1.58+
- Plotly 6.7+
- NumPy 2.2+
- Pandas 2.3+

## 项目结构

```
├── app.py              # Streamlit主应用
├── engine.py           # 工程级计算引擎
├── visualization.py    # 3D可视化组件
├── requirements.txt    # 依赖包
├── .streamlit/         # Streamlit配置
└── README.md           # 项目说明
```

## 计算模型

### 能耗计算
- 鼓风机功率：P = Q × ΔP / η
- 静压头 + 曝气器损失
- 脉冲曝气节能因子

### 剪切力计算
- Shugoll剪切数
- 湍流脉动 + 膜丝振动
- 沿膜丝分布分析

### 污染预测
- Field临界通量理论
- SMP积累模型
- 清洗周期预测

## 许可证

MIT License
