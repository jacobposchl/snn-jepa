# VBNDataHandler requires allensdk (numpy<1.24).  Wrap so the rest of jepsyn.data
# stays importable in environments where allensdk is not installed (e.g. Colab
# on Python 3.12 when only running the training pipeline, not data extraction).
try:
    from .data_handler import VBNDataHandler
except ImportError:
    class VBNDataHandler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VBNDataHandler requires allensdk. "
                "Install it with: pip install allensdk  "
                "(Note: allensdk requires numpy<1.24 and Python ≤3.10)"
            )

from .dataset import REQUIRED_COLUMNS, SpikeWindowDataset, spike_collate_fn
from .preprocess import NeuropixelsPreprocessor, slice_trial_windows, get_or_create_dataset

__all__ = [
    "VBNDataHandler",
    "NeuropixelsPreprocessor",
    "slice_trial_windows",
    "get_or_create_dataset",
    "REQUIRED_COLUMNS",
    "SpikeWindowDataset",
    "spike_collate_fn",
]
