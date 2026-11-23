"""
Error type + handlers so EVERY failure is returned in the contract shape:
    {"error": {"code": ..., "message": ..., "hint": ...}}

The frontend's client.ts unwraps this shape, so nothing may return a bare
FastAPI {"detail": ...} body.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Raise anywhere in the service layer for a clean contract-shaped 4xx."""

    def __init__(self, code: str, message: str, hint: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.hint = hint
        self.status_code = status_code
        super().__init__(message)


def error_body(code: str, message: str, hint: str) -> dict:
    """The one and only error envelope the frontend understands."""
    return {"error": {"code": code, "message": message, "hint": hint}}


def install_error_handlers(app: FastAPI) -> None:
    """Register the three failure paths: APIError, HTTPException, validation."""

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.hint),
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        # Map the status codes we raise deliberately to friendly codes/hints.
        hints = {
            404: "Check the query parameters against the values offered by "
                 "/api/seeds and /api/results/fidelity.",
            422: "Check the request body/parameters against the OpenAPI schema.",
            429: "Wait a moment before retrying; limits are per client IP.",
            503: "A background job is starting; try again shortly.",
        }
        codes = {404: "not_found", 422: "invalid_request", 429: "rate_limited",
                 503: "busy"}
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(
                codes.get(exc.status_code, "error"),
                str(exc.detail),
                hints.get(exc.status_code, "See the API documentation at /docs."),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        return JSONResponse(
            status_code=422,
            content=error_body(
                "invalid_request",
                f"Invalid request: {location or 'body'} — {first.get('msg', 'validation failed')}",
                "Check the request against the OpenAPI schema at /docs.",
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        # Never leak a traceback to the client; log it server-side instead.
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content=error_body(
                "internal_error",
                "Unexpected server error.",
                "Check the API server logs; the request itself was valid.",
            ),
        )
