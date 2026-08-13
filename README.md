# Technical-FAQ

> AI / Computer / Communication 技术问题汇总知识库

本仓库用于每日沉淀技术问答，涵盖 AI、计算机、通信等领域。

## 目录结构

```
Technical-FAQ/
├── README.md               # 项目说明
├── docs/                   # 每日 FAQ 文档（按月份归档）
│   └── 2026-08/
│       └── 2026-08-13-FAQ.md
├── html/                   # HTML 技术文档（按技术领域分类）
│   ├── AI/                 # 人工智能相关
│   ├── Communication/      # 通信网络相关
│   ├── Computer/           # 计算机基础相关
│   └── Other/              # 其他领域
├── templates/              # 模板文件
│   └── daily-faq-template.md
```

## 文档命名规范

- 每日 FAQ 文档：`YYYY-MM-DD-FAQ.md`，存放路径 `docs/YYYY-MM/`
- HTML 技术文档：`YYYY-MM-DD-<原始文件名>.html`，存放路径 `html/<领域>/`

## 内容格式

每篇每日 FAQ 包含：
- 日期
- 问题列表（含分类标签：AI / Computer / Communication / Other）
- 每个问题的简要回答和关键要点
- 信息来源标注

## 更新机制

- 每日自动汇总当天技术问题，生成 FAQ Markdown 文档并推送
- 每日自动扫描新生成的 HTML 技术文档，按领域分类归档并推送
- 按月归档到对应目录
- 定期回顾整理高频问题，形成专题 FAQ
