import ast
import sys
import types
from pathlib import Path

SOURCE_FILE = Path(__file__).parent / "algo.py"


def _extract_string(tree, variable_name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise RuntimeError("algo.py does not define " + variable_name)


def _load_module(name, code):
    module = types.ModuleType(name)
    module.__file__ = "<" + name + ">"
    sys.modules[name] = module
    exec(compile(code, "<" + name + ">", "exec"), module.__dict__)
    return module


def load_algo():
    tree = ast.parse(SOURCE_FILE.read_text(encoding="utf-8"))
    engine = _load_module("contentRankingEngine", _extract_string(tree, "engineSource"))
    api = _load_module("rankingApi", _extract_string(tree, "apiSource"))
    return engine, api
