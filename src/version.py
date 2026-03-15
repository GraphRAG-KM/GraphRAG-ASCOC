from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("GraphRAG-ASCOC")
    except PackageNotFoundError:
        return "0.2.0"
