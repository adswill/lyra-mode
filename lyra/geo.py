from __future__ import annotations

import math

_COUNTRIES: tuple[tuple[str, str, float, float], ...] = (
    ("Afghanistan", "AS", 33.9, 67.7),
    ("Albania", "EU", 41.3, 19.8),
    ("Algeria", "AF", 28.0, 1.7),
    ("Argentina", "SA", -38.4, -63.6),
    ("Armenia", "AS", 40.1, 45.0),
    ("Australia", "OC", -25.3, 133.8),
    ("Austria", "EU", 47.5, 14.6),
    ("Azerbaijan", "AS", 40.1, 47.6),
    ("Bahrain", "AS", 26.0, 50.6),
    ("Bangladesh", "AS", 23.7, 90.4),
    ("Belarus", "EU", 53.7, 27.9),
    ("Belgium", "EU", 50.5, 4.5),
    ("Bolivia", "SA", -16.3, -63.6),
    ("Bosnia", "EU", 43.9, 17.7),
    ("Brazil", "SA", -14.2, -51.9),
    ("Bulgaria", "EU", 42.7, 25.5),
    ("Canada", "NA", 56.1, -106.3),
    ("Chile", "SA", -35.7, -71.5),
    ("China", "AS", 35.9, 104.2),
    ("Colombia", "SA", 4.6, -74.3),
    ("Costa Rica", "NA", 9.7, -83.8),
    ("Croatia", "EU", 45.1, 15.2),
    ("Cuba", "NA", 21.5, -77.8),
    ("Cyprus", "AS", 35.1, 33.4),
    ("Czech Republic", "EU", 49.8, 15.5),
    ("Denmark", "EU", 56.3, 9.5),
    ("Dominican Republic", "NA", 18.7, -70.2),
    ("Ecuador", "SA", -1.8, -78.2),
    ("Egypt", "AF", 26.8, 30.8),
    ("England", "EU", 52.4, -1.2),
    ("Estonia", "EU", 58.6, 25.0),
    ("Ethiopia", "AF", 9.1, 40.5),
    ("Finland", "EU", 61.9, 25.7),
    ("France", "EU", 46.2, 2.2),
    ("Germany", "EU", 51.2, 10.4),
    ("Ghana", "AF", 7.9, -1.0),
    ("Greece", "EU", 39.1, 21.8),
    ("Greenland", "NA", 71.7, -42.6),
    ("Hawaii", "OC", 20.8, -156.3),
    ("Hungary", "EU", 47.2, 19.5),
    ("Iceland", "EU", 64.9, -19.0),
    ("India", "AS", 20.6, 79.0),
    ("Indonesia", "OC", -2.5, 118.0),
    ("Iran", "AS", 32.4, 53.7),
    ("Iraq", "AS", 33.2, 43.7),
    ("Ireland", "EU", 53.1, -8.2),
    ("Israel", "AS", 31.0, 34.9),
    ("Italy", "EU", 41.9, 12.6),
    ("Japan", "AS", 36.2, 138.3),
    ("Jordan", "AS", 30.6, 36.2),
    ("Kazakhstan", "AS", 48.0, 67.0),
    ("Kenya", "AF", -0.0, 37.9),
    ("Kuwait", "AS", 29.3, 47.5),
    ("Latvia", "EU", 56.9, 24.6),
    ("Lebanon", "AS", 33.9, 35.9),
    ("Libya", "AF", 26.3, 17.2),
    ("Lithuania", "EU", 55.2, 23.9),
    ("Luxembourg", "EU", 49.8, 6.1),
    ("Malaysia", "AS", 4.2, 101.9),
    ("Mexico", "NA", 23.6, -102.6),
    ("Moldova", "EU", 47.4, 28.4),
    ("Mongolia", "AS", 46.9, 103.8),
    ("Morocco", "AF", 31.8, -7.1),
    ("Netherlands", "EU", 52.1, 5.3),
    ("New Zealand", "OC", -40.9, 174.9),
    ("Nigeria", "AF", 9.1, 8.7),
    ("North Korea", "AS", 40.3, 127.5),
    ("Northern Ireland", "EU", 54.6, -6.7),
    ("Norway", "EU", 60.5, 8.5),
    ("Oman", "AS", 21.5, 55.9),
    ("Pakistan", "AS", 30.4, 69.3),
    ("Panama", "NA", 8.5, -80.8),
    ("Paraguay", "SA", -23.4, -58.4),
    ("Peru", "SA", -9.2, -75.0),
    ("Philippines", "AS", 12.9, 121.8),
    ("Poland", "EU", 51.9, 19.1),
    ("Portugal", "EU", 39.4, -8.2),
    ("Qatar", "AS", 25.3, 51.2),
    ("Romania", "EU", 45.9, 24.9),
    ("Russia", "EU", 55.8, 37.6),
    ("Saudi Arabia", "AS", 23.9, 45.1),
    ("Scotland", "EU", 56.5, -4.2),
    ("Serbia", "EU", 44.0, 21.0),
    ("Singapore", "AS", 1.4, 103.8),
    ("Slovakia", "EU", 48.7, 19.7),
    ("Slovenia", "EU", 46.2, 14.8),
    ("South Africa", "AF", -30.6, 22.9),
    ("South Korea", "AS", 35.9, 127.8),
    ("Spain", "EU", 40.5, -3.7),
    ("Sri Lanka", "AS", 7.9, 80.8),
    ("Sweden", "EU", 60.1, 18.6),
    ("Switzerland", "EU", 46.8, 8.2),
    ("Syria", "AS", 34.8, 38.9),
    ("Taiwan", "AS", 23.7, 121.0),
    ("Thailand", "AS", 15.9, 100.9),
    ("Tunisia", "AF", 33.9, 9.5),
    ("Turkey", "AS", 38.9, 35.2),
    ("Ukraine", "EU", 48.4, 31.2),
    ("United Arab Emirates", "AS", 23.4, 53.8),
    ("Uruguay", "SA", -32.5, -55.8),
    ("USA", "NA", 39.8, -98.6),
    ("USA", "NA", 41.5, -74.0),
    ("USA", "NA", 34.0, -84.0),
    ("USA", "NA", 41.0, -88.0),
    ("USA", "NA", 45.5, -122.6),
    ("USA", "NA", 34.0, -118.2),
    ("Canada", "NA", 56.1, -106.3),
    ("Canada", "NA", 45.5, -75.7),
    ("Canada", "NA", 49.2, -123.1),
    ("Russia", "EU", 55.8, 37.6),
    ("Russia", "AS", 55.0, 82.9),
    ("Russia", "AS", 62.0, 129.7),
    ("China", "AS", 35.9, 104.2),
    ("China", "AS", 31.2, 121.5),
    ("Australia", "OC", -25.3, 133.8),
    ("Australia", "OC", -33.9, 151.2),
    ("Australia", "OC", -31.9, 115.9),
    ("Brazil", "SA", -14.2, -51.9),
    ("Brazil", "SA", -23.5, -46.6),
    ("Venezuela", "SA", 6.4, -66.6),
    ("Vietnam", "AS", 14.1, 108.3),
    ("Wales", "EU", 52.1, -3.8),
    ("Alaska", "NA", 64.2, -153.4),
    ("Canary Islands", "AF", 28.3, -16.6),
    ("Azores", "EU", 37.7, -25.7),
    ("Madeira", "AF", 32.7, -16.9),
    ("Puerto Rico", "NA", 18.2, -66.5),
    ("Antarctica", "AN", -75.0, 0.0),
)


def grid_to_latlon(grid: str) -> tuple[float, float] | None:
    g = "".join(ch for ch in (grid or "").upper() if ch.isalnum())
    if len(g) < 4 or not g[0].isalpha() or not g[1].isalpha():
        return None
    if not g[2].isdigit() or not g[3].isdigit():
        return None
    lon = (ord(g[0]) - 65) * 20 - 180 + int(g[2]) * 2 + 1.0
    lat = (ord(g[1]) - 65) * 10 - 90 + int(g[3]) * 1 + 0.5
    if len(g) >= 6 and g[4].isalpha() and g[5].isalpha():
        lon += ((ord(g[4]) - 65) + 0.5) / 12.0
        lat += ((ord(g[5]) - 65) + 0.5) / 24.0
        lon -= 1.0
        lat -= 0.5
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def distance_km(grid_a: str, grid_b: str) -> float | None:
    a = grid_to_latlon(grid_a)
    b = grid_to_latlon(grid_b)
    if a is None or b is None:
        return None
    lat1, lon1 = (math.radians(x) for x in a)
    lat2, lon2 = (math.radians(x) for x in b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def lookup(grid: str) -> tuple[str, str]:
    pos = grid_to_latlon(grid)
    if pos is None:
        return "", ""
    lat, lon = pos
    best = _COUNTRIES[0]
    best_d = 1e18
    for name, continent, clat, clon in _COUNTRIES:
        dlat = lat - clat
        dlon = lon - clon
        if dlon > 180:
            dlon -= 360
        elif dlon < -180:
            dlon += 360
        dist = dlat * dlat + dlon * dlon
        if dist < best_d:
            best_d = dist
            best = (name, continent, clat, clon)
    return best[0], best[1]


def dx_text(grid: str, mode: str) -> str:
    if mode in ("", "hidden"):
        return ""
    country, continent = lookup(grid)
    if mode == "country":
        return country
    if mode == "continent":
        return continent
    if country and continent:
        return f"{country}  {continent}"
    return country or continent
