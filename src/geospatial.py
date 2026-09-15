"""Task 7: Core geospatial analysis using actual latitude/longitude/elevation.

Every map/plot answers one of the brief's four questions:
  A. How does temperature vary geographically?
  B. How does rainfall vary geographically?
  C. How does wind speed vary geographically?
  D. Are geographic patterns consistent across years/months?

With only 6 point locations spanning a compact area (lat 5.95-9.66N,
lon 79.86-81.69E), we do NOT attempt spatial interpolation/kriging - that
would fabricate structure between sparse points. Maps here show the actual
observed values at the actual 6 locations only.
"""
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

from config import PROCESSED, TABLES, FIGURES, MAPS, ORIGINAL_6
from plot_style import LOCATION_COLORS, SEQUENTIAL_CMAP

DF_PATH = PROCESSED / "analytical_dataset.parquet"


def load():
    return pd.read_parquet(DF_PATH)


def location_agg(df):
    agg = df.groupby(["location_id", "name", "latitude", "longitude", "elevation_m"]).agg(
        mean_temp=("temperature_2m", "mean"),
        mean_rain=("precipitation", "mean"),
        total_rain_per_year=("precipitation", lambda s: s.sum() / df["year"].nunique()),
        mean_wind=("wind_speed_10m", "mean"),
        mean_humidity=("relative_humidity_2m", "mean"),
    ).reset_index()
    agg.to_csv(TABLES / "task7_location_aggregates.csv", index=False)
    return agg


def static_bubble_maps(agg):
    # Q A/B/C: spatial scatter of temperature / rainfall / wind at the 6 locations.
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    specs = [("mean_temp", "Mean temperature (deg C)", "A: Temperature"),
             ("total_rain_per_year", "Mean annual rainfall (mm)", "B: Rainfall"),
             ("mean_wind", "Mean wind speed (m/s)", "C: Wind speed")]
    for ax, (col, cbar_label, title) in zip(axes, specs):
        sc = ax.scatter(agg.longitude, agg.latitude, c=agg[col], s=400, cmap=SEQUENTIAL_CMAP,
                         edgecolor="black", linewidth=1.2, zorder=3)
        for _, r in agg.iterrows():
            ax.annotate(r["name"], (r.longitude, r.latitude), textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=10)
        ax.set_title(title)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
    fig.suptitle("Spatial distribution of weather variables across the 6 study locations")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_geospatial_bubble_maps.png")
    plt.close(fig)


def lat_lon_relationship_plots(df, agg):
    # Q A/B/C, continued: is there a latitude or longitude gradient?
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for j, (col, ylabel) in enumerate([("mean_temp", "Mean temp (C)"),
                                        ("total_rain_per_year", "Mean annual rainfall (mm)"),
                                        ("mean_wind", "Mean wind speed (m/s)")]):
        for i, (xcol, xlabel) in enumerate([("latitude", "Latitude"), ("longitude", "Longitude")]):
            ax = axes[i, j]
            ax.scatter(agg[xcol], agg[col], c=[LOCATION_COLORS.get(l) for l in agg.location_id], s=150,
                       edgecolor="black", zorder=3)
            for _, r in agg.iterrows():
                ax.annotate(r["name"], (r[xcol], r[col]), textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.suptitle("Temperature / rainfall / wind vs latitude (top) and longitude (bottom)")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_lat_lon_weather_scatter.png")
    plt.close(fig)


def temporal_consistency(df):
    # Q D: is the ranking of locations by temperature/rainfall/wind stable across years?
    yearly_rank = df.groupby(["year", "name"])["temperature_2m"].mean().reset_index()
    yearly_rank["rank"] = yearly_rank.groupby("year")["temperature_2m"].rank(ascending=False)
    pivot = yearly_rank.pivot(index="name", columns="year", values="rank")
    pivot.to_csv(TABLES / "task7_temperature_rank_by_year.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in pivot.index:
        ax.plot(pivot.columns, pivot.loc[name], marker="o", label=name)
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(pivot) + 1))
    ax.set_xlabel("Year"); ax.set_ylabel("Temperature rank (1 = hottest)")
    ax.set_title("D: Is the geographic temperature ranking of locations stable across years?")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "12_temperature_rank_stability.png")
    plt.close(fig)

    print("=== TEMPERATURE RANK BY LOCATION x YEAR (1 = hottest) ===")
    print(pivot.to_string())


def interactive_folium_map(agg):
    center = [agg.latitude.mean(), agg.longitude.mean()]
    m = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")

    vmin, vmax = agg.mean_temp.min(), agg.mean_temp.max()

    def color_for_temp(t):
        # simple linear interpolation blue -> red
        frac = 0 if vmax == vmin else (t - vmin) / (vmax - vmin)
        r = int(255 * frac)
        b = int(255 * (1 - frac))
        return f"#{r:02x}30{b:02x}"

    for _, r in agg.iterrows():
        popup_html = (
            f"<b>{r['name']}</b><br>"
            f"Elevation: {r['elevation_m']:.0f} m<br>"
            f"Mean temperature: {r['mean_temp']:.2f} C<br>"
            f"Mean annual rainfall: {r['total_rain_per_year']:.0f} mm<br>"
            f"Mean wind speed: {r['mean_wind']:.2f} m/s<br>"
            f"Mean humidity: {r['mean_humidity']:.1f} %"
        )
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=10 + (r.mean_temp - vmin) * 4,
            color="black", weight=1, fill=True,
            fill_color=color_for_temp(r.mean_temp), fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=r["name"],
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background:white;
                padding:10px; border:1px solid #999; border-radius:4px; font-size:13px;">
    <b>Marker size & color</b> = mean temperature (2020-2025)<br>
    Click a marker for full location stats
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(MAPS / "sri_lanka_weather_map.html"))
    print(f"Saved interactive map to {MAPS / 'sri_lanka_weather_map.html'}")


def main():
    df = load()
    agg = location_agg(df)
    print("=== LOCATION-LEVEL AGGREGATES (basis for all geospatial maps) ===")
    print(agg.to_string(index=False))
    static_bubble_maps(agg)
    lat_lon_relationship_plots(df, agg)
    temporal_consistency(df)
    interactive_folium_map(agg)
    print("\nGeospatial analysis complete.")


if __name__ == "__main__":
    main()
