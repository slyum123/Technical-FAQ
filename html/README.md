# HTML 技术文档归档

> 按技术领域分类存放的技术分析 HTML 文档

## 目录结构

```
html/
├── AI/              # 人工智能相关（模型架构、训练推理、ML 框架等）
├── Communication/   # 通信网络相关（协议、链路层、Scale-Up 互联等）
├── Computer/        # 计算机基础相关（操作系统、存储、工具链等）
├── Other/           # 其他领域
└── README.md        # 本文件
```

## 文件命名规范

- 格式：`YYYY-MM-DD-<原始文件名>.html`
- 示例：`2026-08-13-transformer-architecture-analysis.html`
- 日期前缀表示文档生成日期，便于按时间检索

## 分类标准

| 领域 | 涵盖范围 |
|------|----------|
| AI | 大模型架构、注意力机制、训练策略、推理优化、强化学习 |
| Communication | 网络协议、链路层流控、GPU 互联、Scale-Up/Scale-Out 架构 |
| Computer | 操作系统、编译器、存储系统、开发工具链 |
| Other | 不属于以上三类的技术文档 |

## 更新机制

- 每日自动扫描工作区新生成的 HTML 文件
- 按内容判断技术领域，复制到对应子目录
- 自动提交并推送到 GitHub
