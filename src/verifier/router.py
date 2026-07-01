"""vr-checker-router — classify an Obligation by the strongest achievable checker.

MVP classification:
- external / IO-boundary leaves -> 'unverifiable' (flagged, not failed).
- everything else with a recovered callable -> 'property_testable'.

The 'formally_checkable' branch (pure/bounded -> SMT) is modeled to preserve the
long-term direction but is dormant in the MVP; vr-formal-checker is deferred.
"""

from .models import Obligation, RoutedObligation


def route(ob: Obligation, *, external: bool = False) -> RoutedObligation:
    if external:
        return RoutedObligation(ob, "unverifiable")
    return RoutedObligation(ob, "property_testable")
