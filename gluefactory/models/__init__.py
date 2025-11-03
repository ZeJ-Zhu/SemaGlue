import importlib.util

from ..utils.tools import get_class
from .base_model import BaseModel


def get_model(name):
    import_paths = [
        name,
        f"{__name__}.{name}",#   三个路径中找
        f"{__name__}.extractors.{name}",  # backward compatibility向后兼容性的路径
        f"{__name__}.matchers.{name}",  # backward compatibility
        f"{__name__}.segments.{name}",
    ]
    for path in import_paths:#遍历路径列表，找到包含模型类的模块
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseModel)#应该是返回类对象
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_model__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Model {name} not found in any of [{" ".join(import_paths)}]')
