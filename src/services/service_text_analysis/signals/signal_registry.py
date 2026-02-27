import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

import structlog

from .indecision_signal import IndecisionSignal
from .reformulation_signal import ReformulationSignal
from .signal_interface import SignalInterface

logger = structlog.get_logger()


def _discover_signals(signals_dir: Optional[Path] = None) -> List[SignalInterface]:
    """Scan signals/ directory for *_signal.py modules that define a SignalInterface subclass."""
    if signals_dir is None:
        signals_dir = Path(__file__).resolve().parent
    instances: List[SignalInterface] = []
    for path in sorted(signals_dir.glob("*_signal.py")):
        name = path.stem
        if name in ("indecision_signal", "reformulation_signal"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr in dir(mod):
                    cls = getattr(mod, attr)
                    if (
                        isinstance(cls, type)
                        and issubclass(cls, SignalInterface)
                        and cls is not SignalInterface
                    ):
                        instances.append(cls())
                        logger.info("signal_registry_discovered", signal_key=instances[-1].key, file=name)
                        break
        except Exception as e:
            logger.warn("signal_registry_scan_skip", file=name, error=str(e))
    return instances


class SignalRegistry:
    """
    Manager: registration and retrieval of active signals.
    Manual list and optional auto-registration by scanning the signals/ directory.
    """

    def __init__(self, auto_scan: bool = False, signals_dir: Optional[Path] = None):
        self.signals: List[SignalInterface] = [
            IndecisionSignal(),
            ReformulationSignal(),
        ]
        if auto_scan:
            self.signals.extend(_discover_signals(signals_dir))

    async def run_all(
        self,
        text: str,
        analyzer: Any,
        context: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run all registered signals with asyncio.gather(); results keyed by signal.key."""
        async def run_one(signal: SignalInterface) -> tuple[str, Dict[str, Any]]:
            try:
                if timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        signal.run(text, analyzer, context),
                        timeout=timeout_seconds,
                    )
                else:
                    result = await signal.run(text, analyzer, context)
                return (signal.key, result)
            except Exception as e:
                logger.warn(
                    "signal_failed",
                    signal_key=signal.key,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return (signal.key, {})

        tasks = [run_one(s) for s in self.signals]
        pairs = await asyncio.gather(*tasks)
        return dict(pairs)