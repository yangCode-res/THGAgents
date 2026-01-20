# Memory/hub.py
from Memory.index import Memory

__all__ = ["get_memory", "reset_memory", "set_memory"]

_memory = None

def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()      # 只在第一次创建
    return _memory

def reset_memory() -> Memory:
    """重置并返回新的全局 Memory，用于批量实验的隔离。"""
    global _memory
    _memory = Memory()
    return _memory

def set_memory(mem: Memory) -> None:
    """显式设置全局 Memory（谨慎使用）。"""
    global _memory
    _memory = mem