"""Generate the DuckDB warehouse the agent queries.

A star schema for container terminal operations: two fact tables surrounded by
conformed dimensions. The data is synthetic but not random - it carries the
patterns a real terminal shows, so analytical questions have answers worth
finding:

* a Q4 peak and a February trough in throughput
* dwell time inflated by hazardous cargo, weekend gate-out and customs holds
* demurrage charged only past the contractual five free days
* berth productivity that scales with crane count, with a per-berth quirk
* operators whose performance is consistently better or worse than average

Deterministic: same seed, same database, so the evaluation suite is
reproducible from a clean clone.

    python scripts/build_warehouse.py
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import date, timedelta
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "port.duckdb"

START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)
SEED = 20260921

FREE_DAYS = 5          # contractual free time before demurrage accrues
DEMURRAGE_PER_DAY = 145.0

OPERATORS = [
    # name, quality multiplier (below 1.0 means faster turnarounds)
    ("Maersk Line", 0.88),
    ("MSC", 0.94),
    ("CMA CGM", 0.97),
    ("Hapag-Lloyd", 0.92),
    ("COSCO Shipping", 1.03),
    ("Evergreen Marine", 1.08),
    ("ONE", 1.00),
    ("Egyptian National Shipping", 1.15),
]

VESSEL_TYPES = [
    ("Feeder", 1200, 2800),
    ("Panamax", 3000, 5100),
    ("Post-Panamax", 5500, 9000),
    ("Neo-Panamax", 9500, 14000),
]

FLAGS = ["Panama", "Liberia", "Marshall Islands", "Malta", "Singapore", "Egypt", "Cyprus"]

BERTHS = [
    # code, terminal, harbour, max draft, cranes
    ("B01", "Alexandria Container Terminal", "West", 11.5, 2),
    ("B02", "Alexandria Container Terminal", "West", 12.0, 3),
    ("B03", "Alexandria Container Terminal", "West", 13.5, 4),
    ("B04", "Alexandria Container Terminal", "West", 14.0, 5),
    ("B05", "El Dekheila Terminal", "West", 14.5, 5),
    ("B06", "El Dekheila Terminal", "West", 15.0, 6),
    ("B07", "El Dekheila Terminal", "West", 13.0, 3),
    ("B08", "East Harbour Quay", "East", 9.0, 1),
]

CARGO_TYPES = [
    # type, hazardous, reefer, dwell multiplier
    ("General Cargo", False, False, 1.00),
    ("Refrigerated", False, True, 0.72),      # reefers are cleared fast
    ("Hazardous Chemicals", True, False, 1.85),
    ("Textiles", False, False, 1.05),
    ("Machinery", False, False, 1.25),
    ("Agricultural Bulk", False, False, 1.40),
    ("Automotive Parts", False, False, 0.95),
]

CUSTOMERS = [
    ("Nile Delta Imports", "Egypt", "Enterprise"),
    ("Alexandria Textiles Co", "Egypt", "Enterprise"),
    ("Mediterranean Freight SA", "Italy", "Mid-Market"),
    ("Levant Trading Group", "Lebanon", "Mid-Market"),
    ("Cairo Industrial Supply", "Egypt", "Enterprise"),
    ("Aegean Logistics", "Greece", "SMB"),
    ("Red Sea Commodities", "Saudi Arabia", "Mid-Market"),
    ("Maghreb Distribution", "Tunisia", "SMB"),
    ("Danube Shipping Partners", "Romania", "SMB"),
    ("Iberian Cargo Solutions", "Spain", "Mid-Market"),
    ("Anatolia Import Export", "Turkey", "Mid-Market"),
    ("Gulf Logistics Holding", "UAE", "Enterprise"),
]


def month_seasonality(month: int) -> float:
    """Throughput multiplier: Q4 peak, February trough."""
    return 1.0 + 0.28 * math.sin((month - 4) * math.pi / 6)


def build_dim_date(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE dim_date AS
        SELECT
            CAST(strftime(d, '%Y%m%d') AS INTEGER)      AS date_key,
            d                                            AS full_date,
            CAST(strftime(d, '%Y') AS INTEGER)           AS year,
            CAST(strftime(d, '%m') AS INTEGER)           AS month,
            strftime(d, '%B')                            AS month_name,
            CAST(quarter(d) AS INTEGER)                  AS quarter,
            CAST(strftime(d, '%d') AS INTEGER)           AS day_of_month,
            dayname(d)                                   AS day_name,
            CASE WHEN dayofweek(d) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
        FROM (
            SELECT UNNEST(
                generate_series(CAST(? AS DATE), CAST(? AS DATE), INTERVAL 1 DAY)
            ) AS d
        )
        """,
        [START_DATE, END_DATE],
    )


def build_dimensions(connection: duckdb.DuckDBPyConnection, rng: random.Random) -> dict:
    connection.execute(
        """
        CREATE TABLE dim_berth (
            berth_key     INTEGER PRIMARY KEY,
            berth_code    VARCHAR,
            terminal      VARCHAR,
            harbour       VARCHAR,
            max_draft_m   DOUBLE,
            crane_count   INTEGER
        )
        """
    )
    for i, (code, terminal, harbour, draft, cranes) in enumerate(BERTHS, start=1):
        connection.execute(
            "INSERT INTO dim_berth VALUES (?, ?, ?, ?, ?, ?)",
            [i, code, terminal, harbour, draft, cranes],
        )

    connection.execute(
        """
        CREATE TABLE dim_cargo_type (
            cargo_type_key INTEGER PRIMARY KEY,
            cargo_type     VARCHAR,
            is_hazardous   BOOLEAN,
            requires_reefer BOOLEAN
        )
        """
    )
    for i, (name, hazardous, reefer, _) in enumerate(CARGO_TYPES, start=1):
        connection.execute(
            "INSERT INTO dim_cargo_type VALUES (?, ?, ?, ?)", [i, name, hazardous, reefer]
        )

    connection.execute(
        """
        CREATE TABLE dim_customer (
            customer_key  INTEGER PRIMARY KEY,
            customer_name VARCHAR,
            country       VARCHAR,
            segment       VARCHAR
        )
        """
    )
    for i, (name, country, segment) in enumerate(CUSTOMERS, start=1):
        connection.execute(
            "INSERT INTO dim_customer VALUES (?, ?, ?, ?)", [i, name, country, segment]
        )

    connection.execute(
        """
        CREATE TABLE dim_vessel (
            vessel_key   INTEGER PRIMARY KEY,
            vessel_name  VARCHAR,
            imo_number   VARCHAR,
            operator     VARCHAR,
            vessel_type  VARCHAR,
            capacity_teu INTEGER,
            flag_country VARCHAR
        )
        """
    )
    prefixes = ["Star", "Ever", "Nile", "Atlas", "Mare", "Orient", "Delta", "Pharos",
                "Aurora", "Cyclade", "Zephyr", "Helios", "Bosphorus", "Cascade"]
    suffixes = ["Trader", "Voyager", "Pioneer", "Spirit", "Horizon", "Explorer",
                "Endeavour", "Runner", "Bridge", "Crest"]

    vessels = []
    used_names: set[str] = set()
    for key in range(1, 61):
        while True:
            name = f"{rng.choice(prefixes)} {rng.choice(suffixes)}"
            if name not in used_names:
                used_names.add(name)
                break
        operator, quality = rng.choice(OPERATORS)
        vessel_type, low, high = rng.choice(VESSEL_TYPES)
        capacity = rng.randint(low, high)
        imo = f"IMO{9000000 + key * 137:07d}"
        connection.execute(
            "INSERT INTO dim_vessel VALUES (?, ?, ?, ?, ?, ?, ?)",
            [key, name, imo, operator, vessel_type, capacity, rng.choice(FLAGS)],
        )
        vessels.append({"key": key, "capacity": capacity, "quality": quality, "type": vessel_type})
    return {"vessels": vessels}


def build_facts(
    connection: duckdb.DuckDBPyConnection, rng: random.Random, vessels: list[dict]
) -> tuple[int, int]:
    connection.execute(
        """
        CREATE TABLE fact_vessel_call (
            call_id            INTEGER PRIMARY KEY,
            vessel_key         INTEGER,
            berth_key          INTEGER,
            arrival_date_key   INTEGER,
            departure_date_key INTEGER,
            waiting_hours      DOUBLE,
            berth_hours        DOUBLE,
            total_moves        INTEGER,
            moves_per_hour     DOUBLE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE fact_container_movement (
            movement_id      INTEGER PRIMARY KEY,
            call_id          INTEGER,
            date_key         INTEGER,
            vessel_key       INTEGER,
            berth_key        INTEGER,
            customer_key     INTEGER,
            cargo_type_key   INTEGER,
            container_count  INTEGER,
            teu              INTEGER,
            dwell_days       DOUBLE,
            customs_hold     BOOLEAN,
            demurrage_usd    DOUBLE
        )
        """
    )

    cargo_multipliers = {i: m for i, (_, _, _, m) in enumerate(CARGO_TYPES, start=1)}
    hazardous_keys = {i for i, (_, h, _, _) in enumerate(CARGO_TYPES, start=1) if h}

    calls: list[tuple] = []
    movements: list[tuple] = []
    call_id = 0
    movement_id = 0

    day = START_DATE
    while day <= END_DATE:
        date_key = int(day.strftime("%Y%m%d"))
        is_weekend = day.weekday() >= 5

        expected = 5.5 * month_seasonality(day.month)
        if is_weekend:
            expected *= 0.55
        arrivals = max(0, int(rng.gauss(expected, 1.4)))

        for _ in range(arrivals):
            call_id += 1
            vessel = rng.choice(vessels)
            # Deep-draft vessels cannot use the shallow East harbour berth.
            candidates = [i for i, b in enumerate(BERTHS, start=1) if not (vessel["capacity"] > 4000 and b[4] <= 2)]
            berth_key = rng.choice(candidates)
            cranes = BERTHS[berth_key - 1][4]

            # Productivity scales with cranes, tempered by operator quality,
            # with a fixed per-berth quirk so one berth is a known laggard.
            berth_quirk = 0.82 if berth_key == 7 else 1.0
            moves_per_hour = max(
                6.0,
                rng.gauss(7.4 * cranes * berth_quirk / vessel["quality"], 2.2),
            )
            total_moves = int(vessel["capacity"] * rng.uniform(0.18, 0.46))
            berth_hours = total_moves / moves_per_hour
            waiting_hours = max(0.0, rng.expovariate(1 / 6.0) * (1.6 if is_weekend else 1.0))

            departure = day + timedelta(hours=waiting_hours + berth_hours)
            departure = min(departure, END_DATE)
            calls.append(
                (
                    call_id,
                    vessel["key"],
                    berth_key,
                    date_key,
                    int(departure.strftime("%Y%m%d")),
                    round(waiting_hours, 2),
                    round(berth_hours, 2),
                    total_moves,
                    round(moves_per_hour, 2),
                )
            )

            # Each call discharges to a handful of customers and cargo types.
            for _ in range(rng.randint(2, 5)):
                movement_id += 1
                cargo_type_key = rng.choices(
                    list(cargo_multipliers), weights=[30, 12, 5, 18, 12, 10, 13]
                )[0]
                customer_key = rng.randint(1, len(CUSTOMERS))
                containers = max(1, int(rng.gauss(total_moves / 8, total_moves / 22)))
                teu = int(containers * rng.uniform(1.25, 1.75))

                customs_hold = rng.random() < (0.22 if cargo_type_key in hazardous_keys else 0.07)
                dwell = rng.gauss(4.6, 1.9) * cargo_multipliers[cargo_type_key]
                if customs_hold:
                    dwell += rng.uniform(2.5, 7.0)
                if is_weekend:
                    dwell += rng.uniform(0.3, 1.2)
                dwell = max(0.5, dwell)

                chargeable = max(0.0, dwell - FREE_DAYS)
                demurrage = round(chargeable * containers * DEMURRAGE_PER_DAY / 20, 2)

                movements.append(
                    (
                        movement_id,
                        call_id,
                        date_key,
                        vessel["key"],
                        berth_key,
                        customer_key,
                        cargo_type_key,
                        containers,
                        teu,
                        round(dwell, 2),
                        customs_hold,
                        demurrage,
                    )
                )
        day += timedelta(days=1)

    connection.executemany(
        "INSERT INTO fact_vessel_call VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", calls
    )
    connection.executemany(
        "INSERT INTO fact_container_movement VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        movements,
    )
    return len(calls), len(movements)


def build(db_path: Path, seed: int = SEED) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(seed)
    connection = duckdb.connect(str(db_path))
    try:
        build_dim_date(connection)
        dimensions = build_dimensions(connection, rng)
        calls, movements = build_facts(connection, rng, dimensions["vessels"])
        connection.execute("CHECKPOINT")
        return {
            "vessel_calls": calls,
            "container_movements": movements,
            "vessels": len(dimensions["vessels"]),
            "berths": len(BERTHS),
            "customers": len(CUSTOMERS),
            "cargo_types": len(CARGO_TYPES),
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stats = build(args.db, args.seed)
    print(f"Built {args.db}")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    print(f"  size: {args.db.stat().st_size / 1_048_576:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
