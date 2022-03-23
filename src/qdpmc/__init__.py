
dependencies = ("numpy", "joblib")
missing_dependencies = []
for dependency in dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{}: {}".format(dependency, e))
        del e
if missing_dependencies:
    raise ImportError(
        "Unable to import required packages: \n" + "\n".join(missing_dependencies)
    )


__version__ = "0.13.1a"

del dependencies, dependency, missing_dependencies

from qdpmc.engine import *
from qdpmc.model import *
from qdpmc.tools import *
from qdpmc.structures import *
from qdpmc.dateutil import Calendar
from qdpmc.products.products import SnowballProd
