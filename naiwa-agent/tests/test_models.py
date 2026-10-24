"""
模型管理器单元测试
==================
不依赖真实 API（构造器被替换为假模型），专注验证：
    - 目录加载与热重载
    - 列表 / 当前模型查询（不含密钥）
    - 切换成功 / 切换失败（不存在、缺密钥、不支持工具、构建失败、ping 失败回滚）
    - 扩展注册（新供应商 / 新模型）
"""
import json

import pytest

from agent.models import (
    ModelBuildError,
    ModelConfigError,
    ModelManager,
    ModelNotFoundError,
    ModelSpec,
    ModelSwitchError,
)


class FakeModel:
    """假聊天模型：记录构造参数，invoke 可被配置成成功或失败。"""
    def __init__(self, spec, key, fail=False):
        self.spec = spec
        self.key = key
        self._fail = fail

    def invoke(self, text):
        if self._fail:
            raise ConnectionError("无法连接模型服务")
        return "pong"


def fake_builder(spec, key):
    return FakeModel(spec, key)


def failing_builder(spec, key):
    raise RuntimeError("构造参数非法")


def make_catalog(tmp_path, models, default=None):
    """生成一个临时 models.json 并返回路径。"""
    data = {"default": default or models[0]["id"], "models": models}
    p = tmp_path / "models.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


BASE_MODELS = [
    {"id": "a-chat", "label": "A对话", "provider": "fake", "model": "a",
     "api_key_env": "A_KEY", "kind": "chat", "supports_tools": True},
    {"id": "a-reasoner", "label": "A推理", "provider": "fake", "model": "a-r",
     "api_key_env": "A_KEY", "kind": "reasoner", "supports_tools": True},
    {"id": "no-tools", "label": "无工具", "provider": "fake", "model": "nt",
     "api_key_env": "A_KEY", "supports_tools": False},
    {"id": "bad-build", "label": "构建失败", "provider": "bad", "model": "x",
     "api_key_env": "A_KEY"},
    {"id": "local", "label": "本地", "provider": "fake", "model": "local",
     "requires_key": False},
]


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("A_KEY", "test-key-123")
    catalog = make_catalog(tmp_path, BASE_MODELS)
    m = ModelManager(catalog)
    m.register_provider("fake", fake_builder)
    m.register_provider("bad", failing_builder)
    return m


# ---- 目录加载 / 热重载 ----
def test_catalog_loads(manager):
    models = manager.list_models()
    assert len(models) == 5
    ids = {m["id"] for m in models}
    assert {"a-chat", "a-reasoner", "no-tools", "bad-build", "local"} <= ids


def test_list_models_no_secret_and_flags(manager):
    models = manager.list_models()
    by_id = {m["id"]: m for m in models}
    # 密钥绝不外泄
    assert all("key" not in json.dumps(m).lower() or "test-key" not in json.dumps(m)
               for m in models)
    # available 标志：有密钥的 True，免密钥的 local 也 True
    assert by_id["a-chat"]["available"] is True
    assert by_id["local"]["available"] is True


def test_reload_picks_up_changes(manager, tmp_path, monkeypatch):
    monkeypatch.setenv("A_KEY", "k")
    catalog = manager._catalog_path
    data = json.loads(catalog.read_text(encoding="utf-8"))
    data["models"].append({"id": "new-one", "label": "新", "provider": "fake",
                           "model": "n", "api_key_env": "A_KEY"})
    catalog.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert manager.reload() == 6
    assert "new-one" in {m["id"] for m in manager.list_models()}


# ---- 切换 ----
def test_switch_success(manager):
    manager.init_default()
    assert manager.current_id == "a-chat"
    info = manager.switch("a-reasoner")
    assert info["current"] is True
    assert manager.current_id == "a-reasoner"
    assert isinstance(manager.current_model, FakeModel)


def test_switch_same_model_is_noop(manager):
    manager.init_default()
    before = manager.current_model
    manager.switch("a-chat")
    assert manager.current_model is before   # 未重建


def test_switch_not_found(manager):
    with pytest.raises(ModelNotFoundError) as exc:
        manager.switch("does-not-exist")
    assert "a-chat" in str(exc.value)   # 提示里列出可用模型


def test_switch_missing_key(manager, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    with pytest.raises(ModelConfigError):
        manager.switch("a-chat")


def test_switch_rejects_no_tool_support(manager):
    with pytest.raises(ModelConfigError):
        manager.switch("no-tools")


def test_switch_build_failure_keeps_old(manager):
    manager.init_default()          # 当前 a-chat
    with pytest.raises(ModelBuildError):
        manager.switch("bad-build")
    # 失败不污染现状
    assert manager.current_id == "a-chat"
    assert isinstance(manager.current_model, FakeModel)


def test_switch_ping_failure_rolls_back(manager, monkeypatch):
    manager.init_default()
    monkeypatch.setattr("agent.models.ModelManager._instantiate",
                        lambda self, spec: FakeModel(spec, "k", fail=True))
    with pytest.raises(ModelSwitchError):
        manager.switch("a-reasoner", ping=True)
    assert manager.current_id == "a-chat"   # 已回滚


# ---- 扩展注册 ----
def test_register_new_provider_and_model(manager):
    manager.register_provider("acme", fake_builder)
    manager.register_model(ModelSpec(
        id="acme-1", label="Acme", provider="acme", model="acme-x",
        api_key_env="A_KEY"))
    info = manager.switch("acme-1")
    assert info["provider"] == "acme"
    assert manager.current_id == "acme-1"
