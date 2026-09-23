from __future__ import annotations

from .feature_leakage import run


def main() -> None:
    outputs = run()
    print(f"[완료] 변수 후보·누수 점검 · {outputs['report']}")


if __name__ == "__main__":
    main()

