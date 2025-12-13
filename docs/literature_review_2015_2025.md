2015-2025 关键Literature Review本整理了1025年间图像加密领域的关键文献，涵盖纯混沌加密、深度学习加密及自适应加密三大方向，旨在明定位价值>**统计概览**:覆盖30+篇（综述类文献覆盖引用数>0），中英文献比例约 4:6。综述与最新进展Survey& Stte-f-the-At这些文献提供了该领域的宏观视野，适合在论文Irc部分引用。*

|年份| 标题 (Ttl)| 作者/期刊 | 主要观点/方向 | 备注 ||:---|---|:---|:---| :--- ||2025| Asecue d effietimageecypschme baed on chaotmp...*| Nature cintifi Repos|纯混沌新型1混沌高维计算复杂问题。 | 代表流派仍在追求“效率”极致 ||2024| *Dee learing-basedimageencryptiontechniquesFundamenls, hallenges,ad future dires | Singh etal./eurocomputing|DL加密综述: 总结了 GA、Autoncoder在加密中应。 | 指出要挑战“可逆性”和“计算开销” ||203** |*Chao-BsdIE:Reiew, Applction,n Chllng* | MDPIMathmatic|混沌综述系统回顾了基于混沌的对称/非对称加密。|强调了混沌与深度学习结合是未来趋势。||2024 |Asuvey f seur emantic communicatos | Guo etal /ScienceDirect|语义通信 关注如何在传输语义特征时保护隐私。 | 与 NSCE的“语义理念高契合 ||2025|Adversarial machinearning:  review of mthos,tools,andchallenges | pringerAIReview | 对抗攻击:综述了针 ML 模型攻击与防御。| 的攻击者模型基础 |.第一类：纯混沌图像加Chaos-basdEcrypto*传统主流方向，强调数学复杂性和统计指标（NPCR/UACI）。*
2.1 典型代表作

| 年份 | 标题 | 核心方法 | 实验结果/局限 |
| :--- | :--- | :--- | :--- |
| 2024**| *A multi-image echemebasd o  nwn-imnonchaoic mdel*Fen et l.)| **n维混沌 + DN编码**: 一次加密多张图，提高吞吐量。 | 结果：PCR>99.6%。<br>局限*:全图均匀加密，无语义感知。|
|2023| * visully meanngfulimagencyption chme based oa 5chaticmp* (Hmthaia et al. | *5D混沌+视觉伪装**:加密图看起来像另一张普通图（隐写）。 | 结果：视觉质量好。<br>局限伪装图容易被 stganalysis识破。 ||2022| *Image encryption agorthm basdnhyprchaot system and pixl scramling|超混沌置乱典的置乱-扩散结构|容易受选择明文 ||2015 |2DSnmodulto mfre (Hua &Zhou)|2D-SLMM经典的混沌系统设计。|地位引用极高的基线方法。|

###2 NSCE 的降维打击传统痛点为了安全，不得不把整张图（含大量无背景都用高维算一遍，慢优势GNN 调度。背景区域只做低维轻量扰动核心区域做高维强加密效率提升显著 3.第二类：深度学习与对抗加密(D & Advrsi)
*新兴方向，强调抗识别能力和生成质量。*

###3.1 典型代表作

| 年份 | 标题 | 核心方法 | 实验结果/局限 |
| :--- | :--- | :--- | :--- |
| **2023** | *Semantc-AwreAdsaraTining(SAAT)* (Yuan al. | 对抗训练*:针对哈希检索任务，通过对抗样本保护隐私。|结果：mAP 下降显著。<br>局限生成图像有噪点，不可逆。|
|**2022** | *Applicaionof mchine earninginintelligentencryption | 参数优化*使用 ML 搜索最优混沌参数。 | 结果：比人工调参快。<br>局限**:离线优化，非实时调度。 |
| ***|DeepEncryptmgEncryptin uing Deep Learning* | **端到端**:使用卷积网络直接输出文|局限缺乏密钥空间定义，安全性难以数证明 |
### 3.2 NSCE 的核心差异
DL痛点生成的图往往**回不去**（不可逆），或者**不安全**（无我们用 DL 做决策，用数学做**执行**既有对抗样本智能，又有学可逆性与安全性4第三类：直接竞品Adaptv/Iget
*这是我们核心赛道，文献较少，属于蓝海。*年份标题核心方法 与对比2024*Deeplearningandchaoticsystembasedimagencptin agorithm* (Jang et al.DL生成密钥:用DL提取图像特征生成混沌初值。还是全图加密，没有区域级调度。2023  *EncryptedSemanticCommunicationUsingAdversarialTraining***语义通信:在特征层进行加密。 | 保护的是特征而非图像，应用场景不同。2019*LearnableImageEncryptionSirichotedumrong et al.可学习ISP:在图像信号处理流程中嵌入加密。针对特定硬件，非通用图像保护。5.我们的NSCE 2025)价值定位

基于上述文献，NSCE 的**独特价值 (Unque SeingPoint** 在于：填补空白**:完美结合了 **TypA(数学安全/可逆和**TypeB(对抗防御/智能)*的优点。
2.**方法创新**:
***可微分调度**: 解决了混沌难用优化的难题
*   **GNN 区域调度**: 真正实现了“像素级”自适应，比全图参数优化更精细3.**性能优势**:     比TyA更快（因为只加密重点区域）。  比TypeB更（因为有学混沌核兜底）## 6 参考文献导出(BibTeXTemplate)
(仅列出部分，完整列表建议使用Zotero/EdNote 导出)*

```bibex
@rtil{sngh2024dep,
 tile={Deep leaning-bsd imae encrption techniques: Fundamentals,challengs, ad futur diecs},author={Singh,P.andetal.},
 journal={eurocomputing},
 year={2024},publisher={Elsevier}
}

@article{mdpi2023chaos,
title={Chaos-BasedImageEncryption:Review, Application, and Challenges},  journal={Mathematics},  volume={11},  year={2023},publisher={MDPI}}
@article{feng4multi,  title={ multi-image encryption scheme based on a new n-dimensional chaoticmodel},  author={Feng, W. and et al.},  journal={os, Solitn&Factals},
 a={2024},
 publish={Elsever}
}
```