try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    if __package__:
        # Inside a package (normal ComfyUI custom-node load): surface the real import
        # error instead of silently falling back to ComfyUI's own top-level `nodes`.
        raise
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
