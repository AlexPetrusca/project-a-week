import time
from ast import List
from dataclasses import dataclass

import requests
import numpy as np
from bs4 import BeautifulSoup


@dataclass
class TableRow:
    x: int
    y: int
    ch: str


def decode_secret_from_google_doc(url: str) -> str:
    raw_page = fetch_url(url)
    page = BeautifulSoup(raw_page, 'html.parser')
    rows = scrape_page(page)
    return decode_secret(rows)


def decode_secret(rows: List[TableRow]) -> str:
    width = max(rows, key=lambda item: item.x).x + 2
    height = max(rows, key=lambda item: item.y).y + 1

    secret = np.full((height, width), ' ', dtype='U1')
    for row in rows:
        secret[height - row.y - 1, row.x] = row.ch
    for y in range(height):
        secret[y, width - 1] = '\n'

    return ''.join(secret.reshape(-1))


def scrape_page(page: BeautifulSoup) -> List[TableRow]:
    parsed_rows = []

    table = page.find('table')
    if table is None:
        raise Exception(f"Failed to scrape page: no table found")

    rows = table.find_all('tr')
    headers = rows[0].find_all('td')
    if len(headers) != 3:
        raise Exception(f"Failed to scrape page: table has wrong number of columns")

    for row in rows[1:]:
        cols = row.find_all('td')
        x = int(cols[0].text)
        y = int(cols[2].text)
        ch = cols[1].text
        parsed_rows.append(TableRow(x, y, ch))

    return parsed_rows


def fetch_url(url: str) -> str:
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }, timeout=10)
        response.raise_for_status()
        return response.text
    except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
        raise Exception(f"Failed to fetch URL '{url}': Caused by ({e})") from e


# URL = "" # invalid url
# URL = "https://dfsdgsgdfhdh.com" # timeout url
# URL = "https://sample.com" # no table
# URL = "https://en.wikipedia.org/wiki/List_of_oldest_living_people" # has table, but wrong number of columns

# URL = 'https://docs.google.com/document/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub'
URL = 'https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9Dlz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub'
print(decode_secret_from_google_doc(URL))
