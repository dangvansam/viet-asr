def declare_namespace(name):
    import sys
    from pkgutil import extend_path
    sys.modules[name].__path__ = extend_path(sys.modules[name].__path__, name)


def get_distribution(package_name):
    import importlib.metadata

    class Distribution:
        def __init__(self, name):
            self.version = importlib.metadata.version(name)
    return Distribution(package_name)


def resource_filename(package_or_requirement, resource_name):
    import os
    import importlib.util
    spec = importlib.util.find_spec(package_or_requirement)
    if spec and spec.submodule_search_locations:
        return os.path.join(spec.submodule_search_locations[0], resource_name)
    return resource_name
