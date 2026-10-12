"""The suite never writes into a developer's instance (CLIM-1395).

Store paths are resolved from the instance config when their module is imported, so conftest
sets the suite's own config before importing anything. This checks that it worked for every
module, including one added later with a store path of its own.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import open_climate_service

from .conftest import TEST_DATA_ROOT


def _import_every_module() -> None:
    for module in pkgutil.walk_packages(open_climate_service.__path__, prefix="open_climate_service."):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue  # an optional dependency this environment lacks; nothing of it to check


def test_every_store_path_resolved_at_import_is_inside_the_test_instance() -> None:
    _import_every_module()
    package = Path(open_climate_service.__file__).parent

    outside = [
        f"{name}.{attr} = {value}"
        for name, module in sorted(sys.modules.items())
        if name == "open_climate_service" or name.startswith("open_climate_service.")
        for attr, value in vars(module).items()
        if isinstance(value, Path)
        and value.is_absolute()
        # Paths into the package's own files (templates, built-in plugins) are not data.
        and not value.resolve().is_relative_to(package)
        and not value.resolve().is_relative_to(TEST_DATA_ROOT.resolve())
    ]

    assert outside == []


def test_the_suite_runs_against_its_own_instance() -> None:
    from open_climate_service import config as api_config

    assert api_config.get_data_root() == TEST_DATA_ROOT.resolve()
