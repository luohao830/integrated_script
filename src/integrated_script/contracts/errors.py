from dataclasses import dataclass, field
from typing import Any, Dict

from ..config.exceptions import ProcessingError


@dataclass
class NormalizedError:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def normalize_exception(error: Exception) -> NormalizedError:
    if isinstance(error, ProcessingError):
        code = getattr(error, "error_code", "PROCESSING_ERROR")
        message = str(getattr(error, "message", "处理失败"))
        details = dict(getattr(error, "context", {}) or {})
        details.setdefault("exception_type", error.__class__.__name__)
        return NormalizedError(code=code, message=message, details=details)

    # 对未知异常统一收敛，避免把内部/敏感细节直接暴露到用户层。
    return NormalizedError(
        code="INTERNAL_ERROR",
        message="内部处理失败",
        details={"exception_type": error.__class__.__name__},
    )
