from open_lars.__version__ import __version__  # noqa
from open_lars.functional import compute_adaptive_lr
from open_lars.lars import LARS

__all__ = ["LARS", "compute_adaptive_lr"]
