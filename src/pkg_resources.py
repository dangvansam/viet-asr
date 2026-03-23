import sys
import os
import importlib.metadata
import importlib.util

# This is a robust shim for pkg_resources to fix compatibility issues with TensorBoard
# when using newer versions of setuptools (>=70) that don't expose pkg_resources.

def declare_namespace(name):
    from pkgutil import extend_path
    if name in sys.modules:
        sys.modules[name].__path__ = extend_path(sys.modules[name].__path__, name)

def get_distribution(package_name):
    class Distribution:
        def __init__(self, name):
            try:
                self.version = importlib.metadata.version(name)
            except Exception:
                self.version = "0.0.0"
    return Distribution(package_name)

def resource_filename(package_or_requirement, resource_name):
    spec = importlib.util.find_spec(package_or_requirement)
    if spec and spec.submodule_search_locations:
        return os.path.join(spec.submodule_search_locations[0], resource_name)
    return resource_name

class EntryPointWrapper:
    def __init__(self, ep):
        self._ep = ep
        self.name = ep.name
        self.dist = None
    def load(self): return self._ep.load()
    def resolve(self): return self._ep.load()
    def __getattr__(self, name): return getattr(self._ep, name)

def iter_entry_points(group, name=None):
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, 'select'):
            group_eps = eps.select(group=group)
        else:
            group_eps = eps.get(group, [])
    except Exception:
        group_eps = []
    
    for ep in group_eps:
        if name is None or ep.name == name:
            yield EntryPointWrapper(ep)

def parse_version(v):
    try:
        from packaging.version import parse
        return parse(str(v))
    except ImportError:
        # Minimal version-like object to satisfy simple comparisons
        class Version:
            def __init__(self, v_str): self.v_str = v_str
            def __str__(self): return self.v_str
            def __repr__(self): return f"Version('{self.v_str}')"
            def __lt__(self, other): return str(self.v_str) < str(other)
        return Version(v)

class ResolutionError(Exception): pass
class DistributionNotFound(ResolutionError): pass
class UnknownExtra(ResolutionError): pass
