"""
tests/test_bridge_uploads.py —— 桥接层附件链路的集成测试
=========================================================
用假 agent / 假模型验证 client/bridge.py 的附件相关行为，不调用真实模型 API：
- upload_file / discard_upload：暂存与丢弃
- send_message + 照片 + 不支持视觉的模型：必须走降级（不调模型、回答含"模型无法识别"）
- send_message + 照片 + 支持视觉的模型：以多模态块（text + image_url）喂给 astream
- send_message + 文本附件：摘录拼进用户消息
- cancel：无任务在跑时安全返回
"""

import asyncio
import base64
import sys
from pathlib import Path

import pytest

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

import bridge  # noqa: E402
from agent import uploads  # noqa: E402
from agent.config import settings  # noqa: E402

PNG_BYTES = bytes([137, 80, 78, 71, 13, 10, 26, 10]) + b"fake-png-body"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """隔离上传目录与会话文件，避免污染项目 data/。"""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(bridge, "SESSION_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(bridge, "_initialized", True)
    yield


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class FakeAI:
    type = "ai"
    tool_calls = None

    def __init__(self, content, msg_id="ai-1"):
        self.content = content
        self.id = msg_id
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class FakeAgent:
    """最小假 agent：记录 astream 收到的输入，吐一条最终答案。"""

    def __init__(self):
        self.captured = {}
        self.called = False

    async def aget_state(self, config):
        return FakeState([FakeAI("历史消息", msg_id="ai-0")])

    def astream(self, input_data, config=None, stream_mode=None):
        self.captured["input"] = input_data
        self.called = True

        async def generator():
            yield ("updates", {"agent": {"messages": [FakeAI("看到图片了", msg_id="ai-2")]}})

        return generator()


def text_model():
    return {"id": "deepseek-chat", "kind": "chat", "supports_vision": False, "pricing": {}}


def vision_model():
    return {"id": "qwen-vl-plus", "kind": "chat", "supports_vision": True, "pricing": {}}


def make_upload(tmp_path, name="photo.png"):
    src = tmp_path / name
    src.write_bytes(PNG_BYTES if name.endswith(".png") else "附件正文内容".encode())
    return bridge.dispatch("upload_file", {"path": str(src), "name": name})


def new_session():
    return bridge.dispatch("new_session", {})


# ---------------------------------------------------------------------------
# 暂存 / 丢弃
# ---------------------------------------------------------------------------
class TestUploadDispatch:
    def test_upload_file_returns_public_view(self, tmp_path):
        result = make_upload(tmp_path)
        assert result["isImage"] is True
        assert result["name"] == "photo.png"
        assert result["uploadId"]
        assert Path(result["path"]).is_file()

    def test_upload_rejects_bad_extension(self, tmp_path):
        src = tmp_path / "evil.exe"
        src.write_bytes(b"MZ")
        with pytest.raises(uploads.UploadError):
            bridge.dispatch("upload_file", {"path": str(src)})

    def test_discard_upload(self, tmp_path):
        upload = make_upload(tmp_path)
        assert bridge.dispatch("discard_upload", {"uploadId": upload["uploadId"]})["discarded"] is True


# ---------------------------------------------------------------------------
# 降级：照片 + 不支持视觉的模型
# ---------------------------------------------------------------------------
class TestVisionShortCircuit:
    def test_image_with_text_model_skips_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)

        def fail_get_agent():
            raise AssertionError("不支持视觉时不应调用模型")

        monkeypatch.setattr(bridge.api, "get_agent", fail_get_agent)

        session = new_session()
        upload = make_upload(tmp_path)
        result = asyncio.run(bridge.send_message({
            "sessionId": session["id"],
            "message": "这张图里有什么",
            "attachments": [{"uploadId": upload["uploadId"], "name": upload["name"],
                             "mime": "image/png", "size": upload["size"]}],
        }))

        assert result["visionUnsupported"] is True
        assert "模型无法识别该照片" in result["answer"]
        assert result["usage"] is None
        # 会话里应同时留下用户消息与降级回答
        stored = bridge.get_session(session["id"])
        assert len(stored["messages"]) == 2
        assert stored["messages"][0]["role"] == "user"
        assert stored["messages"][0]["attachments"][0]["name"] == "photo.png"
        assert stored["messages"][1]["content"] == result["answer"]
        # 照片已归档进会话目录
        assert any(p.suffix == ".png" for p in uploads.session_dir(session["id"]).glob("*"))

    def test_text_only_message_still_calls_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        agent = FakeAgent()
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        result = asyncio.run(bridge.send_message({
            "sessionId": session["id"], "message": "你好", "attachments": [],
        }))
        assert agent.called is True
        assert result["answer"] == "看到图片了"


# ---------------------------------------------------------------------------
# 多模态：照片 + 支持视觉的模型
# ---------------------------------------------------------------------------
class TestMultimodalContent:
    def test_image_sent_as_multimodal_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", vision_model)
        agent = FakeAgent()
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        upload = make_upload(tmp_path)
        asyncio.run(bridge.send_message({
            "sessionId": session["id"],
            "message": "描述这张照片",
            "attachments": [{"uploadId": upload["uploadId"], "name": upload["name"],
                             "mime": "image/png", "size": upload["size"]}],
        }))

        role, content = agent.captured["input"]["messages"][0]
        assert role == "user"
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "描述这张照片"}
        assert content[1]["type"] == "image_url"
        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES

    def test_text_attachment_excerpt_injected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", vision_model)
        agent = FakeAgent()
        monkeypatch.setattr(bridge.api, "get_agent", lambda: agent)

        session = new_session()
        upload = make_upload(tmp_path, name="notes.txt")
        asyncio.run(bridge.send_message({
            "sessionId": session["id"],
            "message": "总结这个文件",
            "attachments": [{"uploadId": upload["uploadId"], "name": "notes.txt",
                             "mime": "text/plain", "size": upload["size"]}],
        }))

        _role, content = agent.captured["input"]["messages"][0]
        assert isinstance(content, str)          # 无图片：纯文本
        assert "总结这个文件" in content
        assert "【附件：notes.txt" in content
        assert "附件正文内容" in content


# ---------------------------------------------------------------------------
# 边界与取消
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_message_rejected(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        session = new_session()
        with pytest.raises(ValueError):
            asyncio.run(bridge.send_message({"sessionId": session["id"], "message": ""}))

    def test_unknown_session_rejected(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        with pytest.raises(ValueError):
            asyncio.run(bridge.send_message({"sessionId": "no-such", "message": "hi"}))

    def test_cancel_when_idle(self):
        assert bridge.dispatch("cancel", {"sessionId": "whatever"}) == {
            "cancelled": False, "reason": "idle"}

    def test_health_reports_timeout(self, monkeypatch):
        monkeypatch.setattr(bridge.api, "current_model", text_model)
        health = bridge.dispatch("health", {})
        assert health["run_timeout_seconds"] == settings.run_timeout_seconds
        assert health["upload_max_mb"] == settings.upload_max_mb
