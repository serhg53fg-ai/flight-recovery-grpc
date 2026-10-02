"""Cutoff-safe parsing and selection for aviation weather reports."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re


UTC = timezone.utc
_ISSUE = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")
_VALIDITY = re.compile(r"\b(\d{2})(\d{2})/(\d{2})(\d{2})\b")


@dataclass(frozen=True)
class WeatherReport:
    kind: str
    airport: str | None
    issue_time: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    source: str
    raw_text: str
    parse_status: str
    features: tuple[tuple[str, float | bool | None], ...]

    def feature_dict(self):
        return dict(self.features)


@dataclass(frozen=True)
class WeatherSnapshot:
    report: WeatherReport | None
    report_age_minutes: float | None
    missing: bool
    stale: bool

    def feature_dict(self):
        values = self.report.feature_dict() if self.report else {}
        return {**values, "report_age_minutes": self.report_age_minutes,
                "weather_missing": self.missing, "weather_stale": self.stale}


def _aware(value, name):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must include timezone")
    return value.astimezone(UTC)


def _previous_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)


def _next_month(year, month):
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _date(year, month, day, hour, minute=0):
    if hour == 24:
        return datetime(year, month, day, tzinfo=UTC) + timedelta(days=1, minutes=minute)
    if not 0 <= hour <= 23:
        raise ValueError("invalid report hour")
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _issue_time(day, hour, minute, reference):
    """Resolve only the explicit reference month or its previous month."""
    candidates = []
    for year, month in ((reference.year, reference.month), _previous_month(reference.year, reference.month)):
        if 1 <= day <= calendar.monthrange(year, month)[1]:
            candidates.append(_date(year, month, day, hour, minute))
    if not candidates:
        raise ValueError("invalid report date")
    return min(candidates, key=lambda value: abs((value - reference).total_seconds()))


def _valid_time(day, hour, anchor, *, after=None):
    candidates = []
    for year, month in ((anchor.year, anchor.month), _next_month(anchor.year, anchor.month)):
        if 1 <= day <= calendar.monthrange(year, month)[1]:
            candidates.append(_date(year, month, day, hour))
    if after is not None:
        later = [value for value in candidates if value > after]
        if later:
            return min(later)
    return min(candidates, key=lambda value: abs((value - anchor).total_seconds()))


def _signed_temperature(value):
    return float(-int(value[1:]) if value.startswith("M") else int(value))


def _features(text, feature_version):
    precipitation_pattern = (r"(?:^|\s)[+-]?(?:TS|SH|FZ)?(?:RA|SN|DZ|SG|PL)(?:\s|$)"
                             if feature_version == 'aviation-weather-v2' else
                             r"(?:^|\s)[+-]?(?:SH|FZ)?(?:RA|SN|DZ|SG|PL)(?:\s|$)")
    values: dict[str, float | bool | None] = {
        "cavok": bool(re.search(r"\bCAVOK\b", text)),
        "ceiling_ft": None,
        "cumulonimbus": bool(re.search(r"\b(?:BKN|OVC)\d{3}CB\b|\bCB\b", text)),
        "dewpoint_c": None,
        "precipitation": bool(re.search(precipitation_pattern, text)),
        "pressure_hpa": None,
        "temperature_c": None,
        "thunderstorm": bool(re.search(r"(?:^|\s)[+-]?TS", text)),
        "visibility_m": None,
        "wind_direction_deg": None,
        "wind_gust_kt": None,
        "wind_speed_kt": None,
    }
    wind = re.search(r"\b(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)\b", text)
    if wind:
        speed_factor = 1.943844 if wind.group(4) == "MPS" else 1.0
        values["wind_direction_deg"] = None if wind.group(1) == "VRB" else float(wind.group(1))
        values["wind_speed_kt"] = float(wind.group(2)) * speed_factor
        values["wind_gust_kt"] = float(wind.group(3)) * speed_factor if wind.group(3) else None
    visibility = re.search(r"(?:^|\s)(\d{4})(?:\s|$)", text)
    if values["cavok"]:
        values["visibility_m"] = 10000.0
    elif visibility:
        values["visibility_m"] = float(visibility.group(1))
    ceilings = [float(item) * 100 for item in re.findall(r"\b(?:BKN|OVC)(\d{3})(?:CB|TCU)?\b", text)]
    if ceilings:
        values["ceiling_ft"] = min(ceilings)
    temperature = re.search(r"\b(M?\d{2})/(M?\d{2})\b", text)
    if temperature:
        values["temperature_c"] = _signed_temperature(temperature.group(1))
        values["dewpoint_c"] = _signed_temperature(temperature.group(2))
    pressure_q = re.search(r"\bQ(\d{4})\b", text)
    pressure_a = re.search(r"\bA(\d{4})\b", text)
    if pressure_q:
        values["pressure_hpa"] = float(pressure_q.group(1))
    elif pressure_a:
        values["pressure_hpa"] = int(pressure_a.group(1)) / 100 * 33.8639
    return tuple(sorted(values.items()))


def parse_weather_report(text, reference_time, source="embedded", *,
                         feature_version='aviation-weather-v1'):
    """Parse METAR or TAF using an explicit timezone-aware date reference."""
    if feature_version not in ('aviation-weather-v1', 'aviation-weather-v2'):
        raise ValueError('unsupported weather feature version')
    reference = _aware(reference_time, "reference_time")
    raw = str(text).strip()
    header = re.match(r"^(METAR|SPECI|TAF)\s+([A-Z0-9]{4})\b", raw)
    kind = "METAR" if header and header.group(1) == "SPECI" else (header.group(1) if header else "UNKNOWN")
    airport = header.group(2) if header else None
    issue_match = _ISSUE.search(raw)
    issue = valid_from = valid_to = None
    status = "OK"
    try:
        if kind not in {"METAR", "TAF"} or issue_match is None:
            raise ValueError("missing weather header or issue time")
        issue = _issue_time(*(int(value) for value in issue_match.groups()), reference)
        if kind == "TAF":
            validity = _VALIDITY.search(raw)
            if validity is None:
                raise ValueError("missing TAF validity")
            start_day, start_hour, end_day, end_hour = map(int, validity.groups())
            valid_from = _valid_time(start_day, start_hour, issue)
            valid_to = _valid_time(end_day, end_hour, valid_from, after=valid_from)
            if valid_to <= valid_from:
                raise ValueError("invalid TAF validity")
    except (ValueError, OverflowError):
        status = "INVALID"
        issue = valid_from = valid_to = None
    current_text = (re.split(r"\s(?:BECMG|TEMPO|RMK)\b", raw, maxsplit=1)[0]
                    if kind == 'METAR' and feature_version == 'aviation-weather-v2' else raw)
    return WeatherReport(kind=kind, airport=airport, issue_time=issue,
                         valid_from=valid_from, valid_to=valid_to, source=str(source),
                         raw_text=raw, parse_status=status,
                         features=_features(current_text, feature_version))


def select_weather_snapshot(reports, cutoff_time, target_time, kind, stale_after_minutes=120):
    cutoff = _aware(cutoff_time, "cutoff_time")
    target = _aware(target_time, "target_time")
    kind = str(kind).upper()
    if kind not in {"METAR", "TAF"}:
        raise ValueError("unsupported weather kind")
    candidates = []
    for report in reports:
        if report.kind != kind or report.parse_status != "OK" or report.issue_time is None:
            continue
        if report.issue_time > cutoff:
            continue
        if kind == "TAF" and not (report.valid_from <= target < report.valid_to):
            continue
        candidates.append(report)
    if not candidates:
        return WeatherSnapshot(None, None, True, False)
    report = max(candidates, key=lambda value: value.issue_time)
    age = (cutoff - report.issue_time).total_seconds() / 60
    return WeatherSnapshot(report, age, False, age > stale_after_minutes)
