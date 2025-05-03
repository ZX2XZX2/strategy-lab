# trading_bot/strategies/__init__.py

import pkgutil
import importlib
from .base import BaseStrategy

def discover_strategies():
    registry = {}
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name in ("base",): continue
        module = importlib.import_module(f"trading_bot.strategies.{module_name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                registry[obj.name] = obj
    return registry
