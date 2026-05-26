"""Dataset utilities for the Chapter 3 hydrogen adsorption experiments.

The maintained preprocessing entry point is
`datasets/custom_data_processor_simplified.py`. Older OCP-style dataset
registrations are treated as legacy code and are not imported here because the
corresponding modules are not present in this cleaned project tree.

Imports are resolved lazily so `import datasets` does not require the full
training/preprocessing environment.
"""

__all__ = [
    "SimpleData",
    "VASPDataProcessor",
]


def __getattr__(name):
    if name in __all__:
        from .custom_data_processor_simplified import SimpleData, VASPDataProcessor

        exports = {
            "SimpleData": SimpleData,
            "VASPDataProcessor": VASPDataProcessor,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
