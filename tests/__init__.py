# -*- coding: utf-8 -*-
"""
Causal-VSE-PC 测试模块

测试文件组织：

Phase 0 - 设计修复与基础设施:
- test_nonce_derivation_unit.py    - 底层nonce派生逻辑测试
- test_deterministic_nonce.py      - 确定性加密API测试
- test_cview_unit.py               - C-view bytes转换逻辑测试
- test_cview_dual_representation.py - C-view完整API测试
- test_aad_binding.py              - AEAD AAD绑定测试
- test_property_encryption_standalone.py - 属性测试（Property 2, 3）

Phase 1 - 数据流水线:
- test_manifest_builder.py         - Manifest构建器测试（Property 4）

其他:
- test_encryption.py               - 基础加密测试
- conftest.py                      - Hypothesis配置和共享fixtures

运行测试:
    .\.venv-scne\Scripts\python.exe tests/<test_file>.py
"""
