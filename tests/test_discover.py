"""
tests/test_discover.py —— 模型自动发现（一家 Key → 这家全部模型）的单元测试
覆盖：正常发现注册 / reasoner 类型识别 / 静态条目不覆盖 / 网络失败缓存回退 / 无 Key 跳过
"""

import json

import pytest

from agent import models


def _manager(tmp_path, entries):
    cat = tmp_path / "models.json"
    cat.write_text(json.dumps({"default": entries[0]["id"],
                               "models": entries}), encoding="utf-8")
    return models.ModelManager(cat)


@pytest.fixture
def manager(tmp_path):
    return _manager(tmp_path, [{
        "id": "base", "label": "Base", "provider": "testprov",
        "model": "base", "base_url": "https://api.testprov.example/v1",
        "api_key_env": "TESTPROV_KEY",
    }])


def test_discover_registers_all_new(manager, monkeypatch, tmp_path):
    monkeypatch.setenv("TESTPROV_KEY", "k")
    monkeypatch.setattr(models, "_fetch_provider_models",
                        lambda base, key, timeout=8: ["base", "flash",
                                                      "v4-pro", "v4-reasoner"])
    report = manager.discover(cache_path=tmp_path / "cache.json")

    assert report["testprov"]["ok"] is True
    assert report["testprov"]["new"] == 3          # base 已存在，不重复注册

    listing = {m["id"]: m for m in manager.list_models()}
    assert set(listing) == {"base", "flash", "v4-pro", "v4-reasoner"}
    assert listing["v4-reasoner"]["kind"] == "reasoner"
    assert listing["flash"]["kind"] == "chat"
    assert listing["flash"]["description"].startswith("自动发现")
    # 自动发现的模型不覆盖静态条目的元数据
    assert listing["base"]["label"] == "Base"


def test_discover_does_not_override_static(manager, monkeypatch, tmp_path):
    monkeypatch.setenv("TESTPROV_KEY", "k")
    monkeypatch.setattr(models, "_fetch_provider_models",
                        lambda base, key, timeout=8: ["base"])
    report = manager.discover(cache_path=tmp_path / "cache.json")
    assert report["testprov"]["new"] == 0
    listing = {m["id"]: m for m in manager.list_models()}
    assert listing["base"]["label"] == "Base"      # 静态优先


def test_discover_no_key_skipped(manager, monkeypatch, tmp_path):
    monkeypatch.delenv("TESTPROV_KEY", raising=False)
    monkeypatch.setattr(models, "_fetch_provider_models",
                        lambda *a, **k: pytest.fail("不应发起请求"))
    report = manager.discover(cache_path=tmp_path / "cache.json")
    assert report["testprov"]["ok"] is False
    assert "未配置密钥" in report["testprov"]["error"]


def test_discover_failure_uses_cache(manager, monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    monkeypatch.setenv("TESTPROV_KEY", "k")

    # 第一次：网络正常，注册并写缓存
    monkeypatch.setattr(models, "_fetch_provider_models",
                        lambda base, key, timeout=8: ["base", "cached-model"])
    r1 = manager.discover(cache_path=cache)
    assert r1["testprov"]["ok"] and r1["testprov"]["new"] == 1

    # 换一个全新 manager 模拟重启，再让网络挂掉 → 应从缓存恢复
    manager2 = _manager(tmp_path, [{
        "id": "base", "label": "Base", "provider": "testprov",
        "model": "base", "base_url": "https://api.testprov.example/v1",
        "api_key_env": "TESTPROV_KEY",
    }])

    def boom(*a, **k):
        raise ConnectionError("network down")

    monkeypatch.setattr(models, "_fetch_provider_models", boom)
    r2 = manager2.discover(cache_path=cache)
    assert r2["testprov"]["ok"] is False

    ids = {m["id"] for m in manager2.list_models()}
    assert "cached-model" in ids


def test_discover_report_failure_shape(manager, monkeypatch, tmp_path):
    monkeypatch.setenv("TESTPROV_KEY", "k")

    def boom(*a, **k):
        raise TimeoutError("slow")

    monkeypatch.setattr(models, "_fetch_provider_models", boom)
    report = manager.discover(cache_path=tmp_path / "cache.json")
    assert report["testprov"]["ok"] is False
    assert "TimeoutError" in report["testprov"]["error"]
    assert "无缓存" in report["testprov"]["error"]
