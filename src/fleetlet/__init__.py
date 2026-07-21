"""fleetlet — Modal/Ray-style functions and actors on smolvm microVMs.

    import fleetlet

    app = fleetlet.App("demo")

    @app.function(workers=4)
    def square(x):
        return x * x

    if __name__ == "__main__":
        with app.run():
            print(square.remote(7))
            print(list(square.map(range(10))))
"""

from ._function import Cls, Function, enter
from .app import App
from .errors import (
    AppNotRunning,
    ConfigError,
    RemoteError,
    FleetletError,
    SmolvmError,
    WorkerError,
)
from .image import Image

__version__ = "0.1.0"

__all__ = [
    "App",
    "Image",
    "enter",
    "Function",
    "Cls",
    "FleetletError",
    "ConfigError",
    "SmolvmError",
    "WorkerError",
    "RemoteError",
    "AppNotRunning",
    "__version__",
]
