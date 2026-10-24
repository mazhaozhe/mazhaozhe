"""tools 包：工具实现与注册表。

导入本包时自动加载全部内置工具，保证 @register 生效。
"""

from tools import coding  # noqa: F401
from tools import documents  # noqa: F401
from tools import search  # noqa: F401
