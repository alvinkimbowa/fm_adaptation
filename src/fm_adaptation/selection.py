from fnmatch import fnmatch


def matches(name, patterns):
    """Empty means keep everything; entries match exactly, as a glob, or as a `_suffix` tag."""
    if not patterns:
        return True
    return any(
        name == pattern or fnmatch(name, pattern) or name.endswith(f"_{pattern}")
        for pattern in patterns
    )
