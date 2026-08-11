"""Small, dependency-free helpers for parsing external timestamps."""

from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime


def parse_external_datetime(value: object) -> datetime | None:
    """Parse an ISO-8601 or RFC 822/2822 timestamp and normalize it to UTC."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None

    parsed: datetime | None = None
    try:
        if len(value) == 10:
            parsed_date = date.fromisoformat(value)
            parsed = datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
        else:
            parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
