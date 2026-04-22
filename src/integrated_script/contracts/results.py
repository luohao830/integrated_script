from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_RESERVED_FIELDS = {"success", "message", "error", "error_code"}


@dataclass
class OperationResult:
    """统一操作结果模型。

    兼容策略：
    - from_legacy: 从旧 dict 构建对象
    - to_legacy: 输出旧 dict 结构，供现有 UI/调用方继续消费
    """

    success: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    error_code: Optional[str] = None
    _legacy: Optional[Dict[str, Any]] = None

    @classmethod
    def from_legacy(cls, legacy: Dict[str, Any]) -> "OperationResult":
        copied = dict(legacy)
        success = bool(copied.get("success", False))

        if success:
            raw_message = copied.get("message", "")
        else:
            raw_error = copied.get("error")
            raw_message = (
                raw_error if raw_error not in (None, "") else copied.get("message", "")
            )

        message = "" if raw_message is None else str(raw_message)
        error_code = copied.get("error_code")
        payload = {
            key: value for key, value in copied.items() if key not in _RESERVED_FIELDS
        }

        return cls(
            success=success,
            payload=payload,
            message=message,
            error_code=error_code,
            _legacy=copied,
        )

    @classmethod
    def failure(
        cls,
        message: str,
        error_code: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "OperationResult":
        return cls(
            success=False,
            payload=dict(payload or {}),
            message=message,
            error_code=error_code,
        )

    def to_legacy(self) -> Dict[str, Any]:
        if self._legacy is not None:
            return dict(self._legacy)

        result: Dict[str, Any] = {
            key: value
            for key, value in self.payload.items()
            if key not in _RESERVED_FIELDS
        }
        result["success"] = self.success

        if self.success:
            if self.message:
                result["message"] = self.message
        else:
            result["error"] = self.message or "处理失败"
            if self.error_code:
                result["error_code"] = self.error_code

        return result
