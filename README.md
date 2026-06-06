# bigdata-churn-prediction
基于 7,043 条电信用户数据，构建端到端用户流失预测系统（Docker 容器化，HDFS + Spark + HBase 全栈）。使用 Spark MLlib Pipeline 构建 45 维特征向量（StringIndexer + OHE + VectorAssembler）；对比逻辑回归、随机森林、GBT 三种模型，AUC 均达 0.83，逻辑回归 F1 最高（0.7985），体现偏差-方差权衡分析能力。预测结果按 RowKey 设计写入 HBase，Streamlit 部署四标签交互式 Dashboard，展示模型对比、特征重要性与统计洞察。
