from importlib.metadata import version as _get_version, PackageNotFoundError

try:
    version = _get_version("axon_velocity")
except PackageNotFoundError:
    version = "0.1.2"
