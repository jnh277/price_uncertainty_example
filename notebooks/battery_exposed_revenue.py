"""
Site Telemetry Analysis

Simplified notebook for loading and analyzing site telemetry data.
"""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datetime import datetime
    from snoopy.application.telemetry.loader import TelemetryLoader
    from snoopy.application.prices.loader import PriceLoader
    import matplotlib.pyplot as plt
    import pandas as pd
    # from snoopy.visualization.telemetry import plot_power_flows, plot_soc
    return PriceLoader, TelemetryLoader, datetime, mo, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Wholesale exposed battery revenue analysis

    Calculate the revenue of a wholesale exposed battery

    ## Configuration

    Set your tenant ID and site ID to get started.
    """)
    return


@app.cell
def _(datetime):
    # Configuration
    tenant_id = "01k5tfkac2hce40z0gkb02yp8q"
    # name = "229 kWh foster pup"
    # site_id = "ffe2c20e-8539-4d96-a1f7-7e152698ea40" # 227kWh virtual battery
    name = "174 kWh foster pup"
    site_id = "eba480d7-88f0-42d3-aab0-e67df5866bfa" # 174 kWh virtual battery
    region = "SA"
    # full period
    start = datetime(2026, 1, 14)
    end = datetime(2026, 1, 30)

    # big spike
    # start = datetime(2026, 1, 26)
    # end = datetime(2026, 1, 28)
    return end, name, region, site_id, start, tenant_id


@app.cell
def _(TelemetryLoader, tenant_id):
    # Initialize the telemetry loader
    loader = TelemetryLoader(tenant_id=tenant_id, cache_dir="./cache")
    return (loader,)


@app.cell
def _(mo):
    mo.md("""
    ## Load Telemetry Data

    Specify the date range for analysis.
    """)
    return


@app.cell
def _(end, loader, site_id, start):
    # Load telemetry data for the specified date range
    # Data is automatically cached as Parquet and loaded into DuckDB
    conn = loader.load_site_telemetry(
        site_id=site_id,
        start_date=start,
        end_date=end,
        force_refresh=False,  # Set to True to re-fetch from database
    )
    return (conn,)


@app.cell
def _(conn, mo, telemetry):
    dft = mo.sql(
        f"""
        SELECT * FROM telemetry
        """,
        engine=conn
    )
    return (dft,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load price data
    """)
    return


@app.cell
def _(PriceLoader, tenant_id):
    # Initialize the price loader
    price_loader = PriceLoader(tenant_id=tenant_id, cache_dir="./cache")
    return (price_loader,)


@app.cell
def _(end, price_loader, region, start):
    # Load AEMO prices for the specified region and date range
    # Data is automatically cached as Parquet and loaded into DuckDB
    price_conn = price_loader.load_aemo_prices(
        region=region,
        start_date=start,
        end_date=end,
        force_refresh=True,  # Set to True to re-fetch from database
    )
    return (price_conn,)


@app.cell
def _(aemo_trading_prices, mo, price_conn):
    prices = mo.sql(
        f"""
        SELECT * FROM aemo_trading_prices
        """,
        engine=price_conn
    )
    return (prices,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Join telemetry and prices
    """)
    return


@app.cell
def _(dft, pd, prices):
    df_all = pd.merge(dft, prices, on='time')
    return (df_all,)


@app.cell
def _(df_all):
    df_all["revenue"] = df_all["battery_energy"] * df_all["rrp"]
    df_all["battery_discharge"] = df_all["battery_energy"].clip(lower=0)
    df_all["battery_charge"] = df_all["battery_energy"].clip(upper=0)

    df_all["charge_revenue"] = df_all["battery_charge"] * df_all["rrp"]
    df_all["discharge_revenue"] = df_all["battery_discharge"] * df_all["rrp"]
    return


@app.cell
def _(df_all, end, name, start):
    total_charge_revenue = df_all["charge_revenue"].sum()
    total_charge_energy = df_all["battery_charge"].sum()
    total_discharge_revenue = df_all["discharge_revenue"].sum()
    total_discharge_energy = df_all["battery_discharge"].sum()

    average_charge_price = total_charge_revenue / total_charge_energy
    average_discharge_price = total_discharge_revenue / total_discharge_energy

    earnings_per_capacity = df_all["revenue"].sum() / float(name[:3])



    print(f"{name} wholesale exposed battery stats for {start} to {end}")
    print("")
    print(f"Total revenue ${df_all["revenue"].sum():.2f}")
    print(f"Earnings per capacity {earnings_per_capacity:.2f} $/kWh")
    print("")

    print(f"Total discharge revenue ${total_discharge_revenue:.2f}")
    print(f"Total discharge energy {total_discharge_energy:.2f} MWh")
    print(f"Average discharging price {average_discharge_price:.2f} $/MWh")
    print("")
    print(f"Total charge revenue ${total_charge_revenue:.2f}")
    print(f"Total charge energy {total_charge_energy:.2f} MWh")
    print(f"Average charging price {average_charge_price:.2f} $/MWh")


    return


@app.cell
def _(df_all, plt):
    plt.figure(figsize=(16, 14))
    plt.subplot(4, 1, 1)
    plt.plot(df_all.time, df_all.battery_energy)
    plt.subplot(4, 1, 2)
    plt.plot(df_all.time, -df_all.battery_energy.cumsum())
    plt.subplot(4, 1, 3)
    plt.plot(df_all.time, df_all.rrp)
    plt.subplot(4, 1, 4)
    # plt.plot(df_all.time, df_all.revenue.cumsum())
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cleanup

    Close connections when done.
    """)
    return


@app.cell
def _(loader):
    # Close database connections
    loader.close()
    return


if __name__ == "__main__":
    app.run()
