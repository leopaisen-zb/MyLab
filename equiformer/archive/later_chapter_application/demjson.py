#!/usr/bin/env python3
"""
demjson兼容性模块，使用demjson3作为后端
"""

try:
    from demjson3 import *
except ImportError:
    # 如果demjson3也不可用，提供基本的JSON功能
    import json
    
    def decode(s, encoding=None):
        return json.loads(s)
    
    def encode(obj, encoding=None):
        return json.dumps(obj)
    
    # 其他可能需要的函数
    loads = json.loads
    dumps = json.dumps 