"""
Streamlit application for monitoring investment funds using local files only.

All data is loaded from Excel or CSV files located in the working directory.  The
application does not rely on any external services or Google Sheets.  The files
expected are:

- ``fund performances.xlsx`` or ``performance_overview.csv`` for performance data.
- ``market views.xlsx`` or ``market_views_filtered.csv`` for market data.
- ``fund exposures.xlsx`` for exposure data.

The views in the app allow filtering and summarisation of these local datasets.
"""

import os
from typing import Dict, List

import pandas as pd
import streamlit as st




# This application exclusively uses local files for data loading.  All
# functionality previously implemented via Google Sheets has been removed.
# The views below read Excel or CSV files from the local working directory.


def load_local_data(file_path: str) -> pd.DataFrame:
    """Load data from a local Excel or CSV file into a DataFrame.

    This helper determines the file type by its extension. Supported
    extensions are .xlsx, .xls and .csv. Raises an exception if the file
    does not exist or is unsupported.

    Parameters
    ----------
    file_path : str
        The path to the file on disk.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the file's contents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    _, ext = os.path.splitext(file_path)
    if ext.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(file_path)
    if ext.lower() == '.csv':
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file extension: {ext}")


def parse_dict(cell: any) -> Dict[str, float]:
    """Convert a dictionary-like cell value into a Python dictionary.

    Values are expected to be of the form "{Key: 12.3%, Other Key: 0.0%}".
    Percent signs are stripped and values are converted to floats when possible.

    Parameters
    ----------
    cell : any
        The cell value from the sheet. May be NaN, dict or string.

    Returns
    -------
    dict
        Parsed dictionary with keys as strings and values as floats or strings.
    """
    if pd.isna(cell):
        return {}
    if isinstance(cell, dict):
        return cell
    text = str(cell).strip("{} ")
    if not text:
        return {}
    result: Dict[str, float] = {}
    for item in text.split(','):
        if not item:
            continue
        if ':' not in item:
            continue
        key, val = item.split(':', 1)
        key = key.strip()
        val = val.strip().replace('%', '')
        try:
            result[key] = float(val)
        except ValueError:
            result[key] = val
    return result


def build_exposure_df(row: pd.Series, prefixes: List[str]) -> pd.DataFrame:
    """Build a DataFrame from a row containing multiple exposure dictionaries.

    Parameters
    ----------
    row : pandas.Series
        The row from the exposures DataFrame.
    prefixes : list[str]
        List of column names containing dictionary strings.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by exposure key with one column per prefix.
    """
    dfs: List[pd.DataFrame] = []
    for col in prefixes:
        dictionary = parse_dict(row.get(col))
        df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[col])
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def show_performance_view() -> None:
    """Render the performance overview page using local data only.

    Loads the performance data from a local file. Supported formats are Excel
    and CSV. Duplicates are removed by Fund Name, Share Class, Currency and
    Date, and unneeded columns are hidden. A fund selector allows filtering.
    """
    # Locate local performance files
    local_path_xlsx = os.path.join(os.getcwd(), "fund performances.xlsx")
    local_path_csv = os.path.join(os.getcwd(), "performance_overview.csv")
    # Load the data
    try:
        if os.path.exists(local_path_xlsx):
            df = pd.read_excel(local_path_xlsx)
        elif os.path.exists(local_path_csv):
            df = pd.read_csv(local_path_csv)
        else:
            st.error("Performance data file not found. Place 'fund performances.xlsx' or 'performance_overview.csv' in the working directory.")
            return
    except Exception as e:
        st.error(f"Failed to load performance data: {e}")
        return
    # Drop duplicates by Fund Name, Share Class, Currency and Date
    dedup_df = df.drop_duplicates(
        subset=[c for c in ["Fund Name", "Share Class", "Currency", "Date"] if c in df.columns],
        keep="last",
    )
    # Columns to hide
    cols_to_hide = [
        "Received",
        "Sender",
        "Category",
        "Net",
        "Gross",
        "Long Exposure",
        "Short Exposure",
        "Correct",
    ]
    display_df = dedup_df.drop(columns=[c for c in cols_to_hide if c in dedup_df.columns])
    # Fund selector
    if "Fund Name" in display_df.columns:
        options = ["All"] + sorted(display_df["Fund Name"].dropna().unique().tolist())
        choice = st.selectbox("Select Fund", options)
        if choice != "All":
            display_df = display_df[display_df["Fund Name"] == choice]
    st.dataframe(display_df, use_container_width=True)


def show_market_view() -> None:
    """Render the market views page using local data only.

    The market views are loaded from a local file ('market views.xlsx' or
    'market_views_filtered.csv'). The Title column is dropped, and the user
    can filter by any column via multi-select lists.
    """
    local_xlsx = os.path.join(os.getcwd(), "market views.xlsx")
    local_csv = os.path.join(os.getcwd(), "market_views_filtered.csv")
    try:
        if os.path.exists(local_xlsx):
            df = pd.read_excel(local_xlsx)
        elif os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
        else:
            st.error("Market views data file not found. Place 'market views.xlsx' or 'market_views_filtered.csv' in the working directory.")
            return
    except Exception as e:
        st.error(f"Failed to load market views data: {e}")
        return
    # Drop Title column if present
    if 'Title' in df.columns:
        df = df.drop(columns=['Title'])
    # Column filter interface
    filter_columns = st.multiselect("Columns to filter", df.columns.tolist())
    filtered_df = df.copy()
    for col in filter_columns:
        choices = sorted(df[col].dropna().unique().tolist())
        selected = st.multiselect(f"{col}", choices)
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]
    st.dataframe(filtered_df, use_container_width=True)


def show_fund_monitor() -> None:
    """Render the fund monitor page using local data only.

    Loads the exposures data from the local ``fund exposures.xlsx`` file.
    Provides selectors for fund and date, displays key metrics, number of
    positions, and detailed sector and geographic exposures.
    """
    # Load local exposures file
    local_path = os.path.join(os.getcwd(), "fund exposures.xlsx")
    try:
        if os.path.exists(local_path):
            df = pd.read_excel(local_path)
        else:
            st.error("Exposure data file not found. Place 'fund exposures.xlsx' in the working directory.")
            return
    except Exception as e:
        st.error(f"Failed to load exposure data: {e}")
        return
    # Ensure date parsing for ordering
    if 'date' in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    # Fund selection
    if 'fund_name' not in df.columns:
        st.error("The exposures file must contain a 'fund_name' column.")
        return
    funds = sorted(df['fund_name'].dropna().unique().tolist())
    # Default fund from secrets if available
    default_fund = st.secrets.get("defaults", {}).get("fund") if hasattr(st, 'secrets') else None
    fund_index = funds.index(default_fund) if default_fund in funds else 0
    fund_choice = st.selectbox("Select Fund", funds, index=fund_index)
    fund_df = df[df['fund_name'] == fund_choice]
    # Date selection
    date_values = fund_df['date'].dropna().unique().tolist()
    date_choice = st.selectbox("Select Date", date_values)
    row = fund_df[fund_df['date'] == date_choice].iloc[0]
    # Metrics display
    metrics_cols = st.columns(5)
    metrics_cols[0].metric("AUM", row.get('aum_fund') or row.get('aum_firm'))
    metrics_cols[1].metric("Net", row.get('net'))
    metrics_cols[2].metric("Gross", row.get('gross'))
    metrics_cols[3].metric("Long", row.get('long'))
    metrics_cols[4].metric("Short", row.get('short'))
    # Positions
    st.subheader("Number of Positions")
    pos_cols = st.columns(2)
    pos_cols[0].metric("Long", row.get('num_pos_long'))
    pos_cols[1].metric("Short", row.get('num_pos_short'))
    # Sector exposures
    sector_keys = ['sector_long', 'sector_short', 'sector_gross', 'sector_net']
    if all(k in row.index for k in sector_keys):
        st.subheader("Sector Exposures")
        sector_df = build_exposure_df(row, sector_keys)
        st.dataframe(sector_df)
    # Geographic exposures
    geo_keys = ['geo_long', 'geo_short', 'geo_gross', 'geo_net']
    if all(k in row.index for k in geo_keys):
        st.subheader("Geographical Exposures")
        geo_df = build_exposure_df(row, geo_keys)
        st.dataframe(geo_df)


def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Fund Monitoring Dashboard", layout="wide")
    # Determine which page to display
    page = st.sidebar.radio("Select View", [
        "Performance Overview",
        "Market Views",
        "Fund Monitor",
    ])
    if page == "Performance Overview":
        show_performance_view()
    elif page == "Market Views":
        show_market_view()
    elif page == "Fund Monitor":
        show_fund_monitor()


if __name__ == "__main__":
    main()