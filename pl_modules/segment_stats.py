"""Модуль для вычисления статистик сегментов ЭМГ данных."""

from pathlib import Path

from pl_modules.data import load_and_segment


if __name__ == "__main__":
    # Путь к папке с .mat файлами (как в data.py)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    print(f"DATA_ROOT: {DATA_ROOT}")
    load_and_segment(str(DATA_ROOT))
