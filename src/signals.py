# src/signals.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class TradingSignal:
    signal: str
    risk_label: str
    confidence_label: str
    confidence_score: float
    predicted_return_pct: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _base_signal(predicted_return_pct: float) -> str:
    if predicted_return_pct > 1.0:
        return "PaperTrackCandidate"
    if 0.2 <= predicted_return_pct <= 1.0:
        return "WatchlistOnly"
    if -0.2 < predicted_return_pct < 0.2:
        return "NeutralResearch"
    if -1.0 <= predicted_return_pct <= -0.2:
        return "RejectedForNow"
    return "HighRiskResearchOnly"


def _risk_label(predicted_return_pct: float, range_width_pct: Optional[float]) -> str:
    abs_ret = abs(predicted_return_pct)

    if range_width_pct is None:
        if abs_ret < 0.2:
            return "Low"
        if abs_ret <= 1.0:
            return "Medium"
        return "High"

    if range_width_pct > 4.0:
        return "High"
    if range_width_pct > 2.0:
        return "Medium"

    if abs_ret > 1.0:
        return "Medium"
    return "Low"


def _confidence_score(
    predicted_return_pct: float,
    lower_return_pct: Optional[float],
    upper_return_pct: Optional[float],
) -> float:
    """
    Simple confidence score:
    - Higher when prediction range does not cross zero.
    - Lower when range is too wide relative to expected return.
    """

    abs_ret = abs(predicted_return_pct)

    if lower_return_pct is None or upper_return_pct is None:
        if abs_ret > 1.0:
            return 70.0
        if abs_ret >= 0.2:
            return 55.0
        return 40.0

    range_width = abs(upper_return_pct - lower_return_pct)
    crosses_zero = lower_return_pct <= 0 <= upper_return_pct

    if crosses_zero:
        base = 35.0
    else:
        base = 65.0

    if abs_ret > 1.0:
        base += 15.0
    elif abs_ret >= 0.2:
        base += 5.0

    if range_width > 4.0:
        base -= 20.0
    elif range_width > 2.0:
        base -= 10.0
    elif range_width < 1.0:
        base += 10.0

    return float(np.clip(base, 5.0, 95.0))


def _confidence_label(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def generate_trading_signal(
    *,
    predicted_return_pct: float,
    lower_return_pct: Optional[float] = None,
    upper_return_pct: Optional[float] = None,
) -> TradingSignal:
    try:
        predicted_return_pct = float(predicted_return_pct)
    except Exception as exc:
        raise ValueError("predicted_return_pct must be numeric.") from exc

    if not np.isfinite(predicted_return_pct):
        raise ValueError("predicted_return_pct must be finite.")

    signal = _base_signal(predicted_return_pct)

    range_width_pct = None
    if lower_return_pct is not None and upper_return_pct is not None:
        lower_return_pct = float(lower_return_pct)
        upper_return_pct = float(upper_return_pct)
        range_width_pct = abs(upper_return_pct - lower_return_pct)

    risk_label = _risk_label(predicted_return_pct, range_width_pct)

    score = _confidence_score(
        predicted_return_pct=predicted_return_pct,
        lower_return_pct=lower_return_pct,
        upper_return_pct=upper_return_pct,
    )

    confidence_label = _confidence_label(score)

    explanation = (
        f"Predicted return is {predicted_return_pct:.2f}%. "
        f"Signal rule gives '{signal}'. "
        f"Confidence is {confidence_label.lower()} because the prediction range "
        f"{'was considered' if range_width_pct is not None else 'was not available'}."
    )

    return TradingSignal(
        signal=signal,
        risk_label=risk_label,
        confidence_label=confidence_label,
        confidence_score=round(score, 2),
        predicted_return_pct=round(predicted_return_pct, 4),
        explanation=explanation,
    )


if __name__ == "__main__":
    examples = [1.94, 0.6, 0.05, -0.5, -1.4]

    for value in examples:
        signal = generate_trading_signal(
            predicted_return_pct=value,
            lower_return_pct=value - 0.8,
            upper_return_pct=value + 0.8,
        )
        print(signal.to_dict())
