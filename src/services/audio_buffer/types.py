from typing import Awaitable, Callable, Tuple

AudioKey = Tuple[str, str, str]
BufferReadyCallback = Callable[..., Awaitable[None]]

