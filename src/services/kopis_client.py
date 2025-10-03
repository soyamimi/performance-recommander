from datetime import datetime, timedelta
from typing import Any

import requests
import xmltodict
import pandas as pd

from conf.config import settings


class KopisClient:
    def __init__(self):
        self.base_url = settings.API_BASE_URL.rstrip("/")
        self.api_key = settings.API_KEY
        self.timeout = settings.REQUEST_TIMEOUT

    @staticmethod
    def split_date_range(
        start_date: str, end_date: str, chunk_days: int = 30
    ) -> list[Any]:
        start = datetime.strptime(start_date, "%d%m%Y")
        end = datetime.strptime(end_date, "%d%m%Y")
        ranges = []

        while start <= end:
            chunk_end = start + timedelta(days=chunk_days - 1)
            if chunk_end > end:
                chunk_end = end
            ranges.append((start.strftime("%d%m%Y"), chunk_end.strftime("%d%m%Y")))
            start = chunk_end + timedelta(days=1)
        return ranges

    def fetch_performance_basic_info(
        self, start_date: str, end_date: str, page: int = 1, rows: int = 100
    ) -> dict[str, Any]:
        url = f"{self.base_url}/pblprfr"
        params = {
            "service": self.api_key,
            "stdate": start_date,
            "eddate": end_date,
            "cpage": str(page),
            "rows": str(rows),
        }
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = xmltodict.parse(response.text)
        return data

    def fetch_period_data(self, start_date: str, end_date: str) -> list[Any]:
        all_records = []
        page = 1
        while True:
            data = self.fetch_performance_basic_info(start_date, end_date, page, 100)
            records = data.get("dbs", {}).get("db", [])
            if not records:
                break
            if isinstance(records, dict):
                records = [records]
            all_records.extend(records)
            if len(records) < 100:
                break
            page += 1
        return all_records

    def fetch_performances(self, year_start: str, year_end: str) -> pd.DataFrame:
        date_ranges = self.split_date_range(year_start, year_end, 30)

        all_records = []
        for start, end in date_ranges:
            print(f"Fetching {start} ~ {end}")
            period_records = self.fetch_period_data(start, end)
            all_records.extend(period_records)

        return pd.DataFrame(all_records)

    def fetch_performance_detail(self, id_performance: str) -> list[Any] | None:

        url = f"{self.base_url}/pblprfr/{id_performance}"
        params = {"service": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = xmltodict.parse(response.text)
            return data.get("dbs", {}).get("db", [])
        except Exception as e:
            print(f"[ERROR] Failed to fetch {id_performance}: {e}")
            return None
