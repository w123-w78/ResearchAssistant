# -*- coding: utf-8 -*-
"""
配置管理模块
------------
统一从 config.yaml 读取配置，支持：
  1. 点号路径取值  config.get("llm.temperature")
  2. 环境变量覆盖  APP_LLM__PROVIDER=ollama
  3. 运行时热更新  config.set("retrieval.top_k", 10)

设计目的：所有实验参数（chunk_size / top_k / 路由阈值 ...）集中在一处，
模块五做参数对比实验时只需改 yaml，无需改代码。
"""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .env import PROJECT_ROOT, load_dotenv  # noqa: F401  (PROJECT_ROOT 用于路径回退)

# 需要锚定到项目根目录的路径型配置（相对路径不再随终端 cwd 漂移）
_PATH_KEYS = (
    "app.data_dir", "app.upload_dir", "app.index_dir", "app.log_dir",
    "storage.db_path", "vector_store.index_path", "logging.file",
    "observability.request_log_path", "cache.persist_path",
)


def _anchor_paths(data: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """把相对路径配置锚定到项目根目录，避免结果随运行目录变化。"""
    for key in _PATH_KEYS:
        node: Any = data
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                node = None
                break
            node = node[p]
        if not isinstance(node, dict):
            continue
        val = node.get(parts[-1])
        if isinstance(val, str) and val and not Path(val).is_absolute():
            node[parts[-1]] = str((root / val).resolve())
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 优先。"""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """轻量配置容器（避免引入 pydantic 依赖，保持核心链路零负担）。"""

    _instance: Optional["Config"] = None
    _lock = threading.Lock()

    def __init__(self, data: Dict[str, Any], config_path: Optional[str] = None):
        self._data: Dict[str, Any] = data or {}
        self.config_path = config_path
        self._apply_env_overrides()

    # ---------------- 构造 ----------------
    @classmethod
    def load(cls, path: str = "config.yaml", reload: bool = False) -> "Config":
        """加载配置文件（默认单例，reload=True 强制重新读取）。"""
        with cls._lock:
            if cls._instance is not None and not reload:
                return cls._instance
            # 先加载 .env（如 DASHSCOPE_API_KEY），再读配置
            load_dotenv()
            cfg_path = Path(path) if path else (PROJECT_ROOT / "config.yaml")
            if not cfg_path.exists():
                # 相对路径在当前工作目录下找不到时，回退到项目根目录。
                # 场景：IDE 直接运行 src/frontend/app.py，cwd 是 frontend 目录，
                # 此时必须仍能定位到项目根的 config.yaml。
                alt = PROJECT_ROOT / Path(path).name
                if alt.exists():
                    cfg_path = alt
            data: Dict[str, Any] = {}
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                # 配置文件缺失时不崩溃，使用内置默认值（便于单元测试）
                data = _DEFAULT_CONFIG
            # 允许 config.local.yaml 覆盖（本地私密配置不入库）
            local_path = cfg_path.parent / "config.local.yaml"
            if local_path.exists():
                with open(local_path, "r", encoding="utf-8") as f:
                    data = _deep_merge(data, yaml.safe_load(f) or {})
            # 相对路径统一锚定到配置文件所在目录（即项目根）
            data = _anchor_paths(data, cfg_path.parent.resolve())
            cls._instance = cls(data, str(cfg_path))
            return cls._instance

    @classmethod
    def instance(cls) -> "Config":
        """获取已加载的配置实例；未加载则尝试默认路径。"""
        if cls._instance is None:
            return cls.load()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ---------------- 取值 ----------------
    def get(self, key: str, default: Any = None) -> Any:
        """按点号路径取值：get("retrieval.top_k") -> 20"""
        node: Any = self._data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key: str, value: Any) -> None:
        """运行时修改配置（实验调参用）。"""
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def section(self, name: str) -> Dict[str, Any]:
        """返回某个配置段落的副本。"""
        return copy.deepcopy(self.get(name, {}) or {})

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    # ---------------- 环境变量覆盖 ----------------
    def _apply_env_overrides(self) -> None:
        """
        支持两类环境变量：
          APP_LLM__PROVIDER=ollama   -> llm.provider
          APP_LLM_MODEL=qwen2.5:14b  -> llm.model
        便于 Docker / CI 中注入，无需改文件。
        """
        for env_key, env_val in os.environ.items():
            if not env_key.startswith("APP_"):
                continue
            body = env_key[4:]
            if "__" in body:
                path = body.lower().replace("__", ".")
            else:
                parts = body.lower().split("_", 1)
                path = parts[0] + "." + parts[1] if len(parts) == 2 else body.lower()
            if self.get(path, "__missing__") != "__missing__":
                self.set(path, _cast(env_val, type(self.get(path))))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(path={self.config_path}, sections={list(self._data.keys())})"


def _cast(value: str, target_type: type) -> Any:
    """把字符串环境变量转成目标类型。"""
    try:
        if target_type is bool:
            return value.strip().lower() in ("1", "true", "yes", "on")
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
    except (ValueError, TypeError):
        pass
    return value


# 配置文件缺失时的兜底默认值
_DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {"name": "智能科研助理", "data_dir": "./data", "index_dir": "./data/index",
            "upload_dir": "./data/uploads", "log_dir": "./logs"},
    "llm": {"model": "qwen-plus", "temperature": 0.2, "top_p": 0.85,
            "max_tokens": 2048, "timeout": 120, "max_retries": 2,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": ""},
    "embedding": {"model": "text-embedding-v3", "dim": 1024, "batch_size": 10,
                  "normalize": True, "timeout": 60, "max_retries": 2},
    "storage": {"db_path": "./data/assistant.db", "enable_request_log": True,
                "enable_cache": True},
    "chunking": {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 64,
                 "min_chunk_size": 80},
    "retrieval": {"top_k": 20, "final_top_k": 5, "rrf_k": 60, "vector_weight": 0.5,
                  "bm25_weight": 0.5, "fusion_mode": "rrf", "score_threshold": 0.15,
                  "enable_rerank": True, "reranker": {"provider": "dashscope"}},
    "generation": {"stream": True, "max_context_chars": 6000, "temperature": 0.2,
                   "citation_style": "inline", "enable_citation": True},
    "cache": {"enabled": True, "similarity_threshold": 0.93, "max_size": 500,
              "ttl_seconds": 86400},
    "agent": {"max_iterations": 6, "tool_timeout": 30, "max_tool_retries": 2,
              "enable_router": True, "enable_parallel": True, "max_tool_workers": 4,
              "loop_detect_window": 3},
    "memory": {"max_turns": 20, "max_tokens": 3000, "summarize_threshold": 8},
    "logging": {"level": "INFO", "console": True, "file": "./logs/app.log"},
}
