# -*- coding: utf-8 -*-
"""
环境变量加载（.env 文件支持）
------------------------------
为什么自己实现而不装 python-dotenv：
  少一个依赖，且本项目只需要最基础的 KEY=VALUE 解析。

加载规则：
  1. 从项目根目录读取 .env（config.yaml 同级）
  2. 忽略 # 开头的注释行与空行
  3. 自动去除首尾空格与成对引号
  4. **不覆盖** 已存在的系统环境变量（系统变量优先级更高）
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录：src/utils/env.py -> 上三级
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LOADED = False


def find_dotenv(start: Path | None = None) -> Path | None:
    """从项目根目录向上查找 .env 文件。"""
    cur = start or PROJECT_ROOT
    for _ in range(4):
        candidate = cur / ".env"
        if candidate.exists():
            return candidate
        cur = cur.parent
    return None


def load_dotenv(path: str | None = None, override: bool = False) -> dict:
    """
    加载 .env 到 os.environ，返回本次实际写入的键值（值已脱敏）。

    :param path: 指定 .env 路径；默认自动查找
    :param override: 是否覆盖已存在的环境变量（默认否）
    """
    global _LOADED
    env_path = Path(path) if path else find_dotenv()
    loaded: dict = {}
    if env_path is None or not env_path.exists():
        _LOADED = True
        return loaded

    with open(env_path, "r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            # 去掉行内注释（值不含 # 的简单场景，API Key 不会含 #）
            if " #" in line:
                line = line.split(" #", 1)[0].strip()
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value
                loaded[key] = _mask(value)
    _LOADED = True
    return loaded


def _mask(value: str) -> str:
    """脱敏：只保留前 6 位，用于日志确认 Key 已加载。"""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:6]}{'*' * (len(value) - 6)}"


def ensure_loaded() -> None:
    """确保 .env 只加载一次（幂等）。"""
    if not _LOADED:
        load_dotenv()


def get_api_key(*names: str) -> str:
    """按优先级取第一个非空 Key，并做明显占位符过滤。"""
    placeholders = {"", "EMPTY", "your-api-key", "sk-xxxxxxxxxxxxxxxxxxxxxxxx",
                    "sk-xxxxxx", "changeme"}
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v and v not in placeholders:
            return v
    return ""
