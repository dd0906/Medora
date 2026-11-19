"""Thin client for the Medora REST API."""

from typing import Any

import httpx

PREFIX = "/api/v1"


class ApiUnavailable(RuntimeError):
    """The API couldn't be reached."""


class ApiError(RuntimeError):
    """The API refused a request. The message is safe to show to the user."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class MedoraClient:
    def __init__(self, http: httpx.Client, api_key: str | None = None) -> None:
        self._http = http
        self._headers = {"X-API-Key": api_key} if api_key else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, f"{PREFIX}{path}", headers=self._headers, **kwargs)
        except httpx.TransportError as error:
            raise ApiUnavailable("The Medora API couldn't be reached.") from error

        if response.is_error:
            raise ApiError(response.status_code, _error_message(response))
        return response.json() if response.content else None

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def statistics(self, source: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/stats", params=_params(source=source))

    def timeline(self, source: str | None = None, bucket: str = "second") -> list[dict[str, Any]]:
        return self._request("GET", "/stats/timeline", params=_params(source=source, bucket=bucket))

    def label_outcomes(self, source: str | None = None) -> dict[str, list[dict[str, Any]]]:
        return self._request("GET", "/stats/outcomes", params=_params(source=source))

    def events(self, source: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._request("GET", "/events", params=_params(source=source, limit=limit))

    def alerts(self, **filters: Any) -> list[dict[str, Any]]:
        return self._request("GET", "/alerts", params=_params(**filters))

    def alert(self, alert_id: str) -> dict[str, Any]:
        return self._request("GET", f"/alerts/{alert_id}")

    def alert_outcomes(self, alert_id: str) -> dict[str, int]:
        return self._request("GET", f"/alerts/{alert_id}/outcomes")

    def update_alert(self, alert_id: str, status: str) -> dict[str, Any]:
        return self._request("PATCH", f"/alerts/{alert_id}", json={"status": status})

    def model(self) -> dict[str, Any]:
        return self._request("GET", "/model")

    def evaluation(self) -> dict[str, Any]:
        return self._request("GET", "/evaluation")

    def replay_status(self) -> dict[str, Any]:
        return self._request("GET", "/replay")

    def start_replay(self, limit: int | None, rate: float | None) -> dict[str, Any]:
        return self._request("POST", "/replay", json={"limit": limit, "rate": rate})

    def stop_replay(self) -> dict[str, Any]:
        return self._request("POST", "/replay/stop")

    def clear_replay_data(self) -> None:
        self._request("DELETE", "/replay/data")

    def clear_rule_findings(self) -> None:
        self._request("DELETE", "/rule-findings")


def _params(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"The API returned an error ({response.status_code})."
