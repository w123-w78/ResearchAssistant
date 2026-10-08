# -*- coding: utf-8 -*-
"""存储层：SQLite 持久化（文档元数据 / 对话记忆 / 语义缓存 / 请求日志）。"""
from .database import Database, get_database

__all__ = ["Database", "get_database"]
