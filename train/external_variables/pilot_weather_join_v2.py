"""Province-mapping supplement for the domestic weather pilot."""

from __future__ import annotations

from . import pilot_weather_join as base


base.PROVINCE_ALIASES.update(
    {
        "영주시": "경상북도",
        "봉화군": "경상북도",
        "안동시": "경상북도",
        "상주시": "경상북도",
        "의성군": "경상북도",
        "청도군": "경상북도",
        "구미시": "경상북도",
        "성주군": "경상북도",
        "서귀포시": "제주특별자치도",
        "수원시": "경기도",
        "나주시": "전라남도",
        "전라븍도": "전북특별자치도",
    }
)


if __name__ == "__main__":
    base.main()
