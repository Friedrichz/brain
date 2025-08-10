"""
Streamlit application for monitoring investment funds using data
directly from Google Sheets.

This version of the app relies exclusively on the Google Sheets
API via the `gspread` library.  To run the app you must supply
valid credentials for a Google service account either through
Streamlit's secrets mechanism or by setting the
``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to point
at your downloaded JSON key file.  See the documentation below
for details.

Structure
---------

* Performance Overview – fetches the ``fund performances`` sheet, hides
  specified columns, deduplicates by Fund Name/Share Class/Currency/Date
  and allows the user to filter by fund.
* Market Views – fetches the ``market views`` sheet, hides the
  ``Title`` column and provides multi‑select filters for all
  remaining columns.
* Fund Monitor – fetches the ``fund exposures`` sheet, allows the
  user to select a fund and date, displays key metrics (AUM,
  net/gross/long/short exposures), counts of positions and
  tabular sector/geographical exposures by unpacking nested
  dictionaries.

To configure which Google Sheets to read from, set the following

or in the "Secrets" section of your Streamlit Cloud app:

```
[fund_performances]
sheet_id = "YOUR_FUND_PERFORMANCES_SHEET_ID"
worksheet = "Sheet1"

[market_views]
sheet_id = "YOUR_MARKET_VIEWS_SHEET_ID"
worksheet = "Sheet1"

[exposures]
sheet_id = "YOUR_FUND_EXPOSURES_SHEET_ID"
worksheet = "Sheet1"

[defaults]
fund = "Helikon Long Short Equity Fund"  # optional default fund

```

The ``gcp_service_account`` section should contain the exact
contents of the JSON key file downloaded from Google Cloud (line
breaks in the private key must be escaped with ``\n``).  This file
should never be committed to source control.  Instead, use
``st.secrets`` locally and copy the same contents into the
Streamlit Cloud app's secrets when you deploy.
"""

import os
from typing import Dict, List

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import gspread
from st_aggrid import AgGrid, GridOptionsBuilder


def get_gspread_client() -> gspread.Client:
    """Return an authenticated gspread client.

    The function attempts to authenticate using credentials stored in
    Streamlit secrets under ``gcp_service_account``.  If those are not
    present, it falls back to the standard environment variable
    ``GOOGLE_APPLICATION_CREDENTIALS`` pointing at a JSON key file.

    Raises
    ------
    RuntimeError
        If no credentials can be found.
    """
    # First try to load credentials from Streamlit secrets
    if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        try:
            client = gspread.service_account_from_dict(creds_dict)
            return client
        except Exception as exc:
            st.error(f"Failed to authenticate using gcp_service_account in secrets: {exc}")
            raise
    # Fallback to environment variable
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        try:
            client = gspread.service_account(filename=creds_path)
            return client
        except Exception as exc:
            st.error(f"Failed to authenticate using GOOGLE_APPLICATION_CREDENTIALS at {creds_path}: {exc}")
            raise
    # If we reach here, no credentials were found
    raise RuntimeError(
        "Google Sheets credentials not found. Provide a service account key via "
        "st.secrets['gcp_service_account'] or set the GOOGLE_APPLICATION_CREDENTIALS environment variable."
    )


def load_sheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    """Load a Google Sheet into a DataFrame.

    Parameters
    ----------
    sheet_id : str
        The ID of the spreadsheet (the part between ``/d/`` and ``/edit`` in
        the Google Sheets URL).
    worksheet : str
        The name of the worksheet/tab within the spreadsheet.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the sheet's rows.
    """
    client = get_gspread_client()
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(worksheet)
        records = ws.get_all_records()
    except Exception as exc:
        st.error(f"Failed to load sheet {sheet_id}/{worksheet}: {exc}")
        raise
    return pd.DataFrame(records)


def parse_dict(cell: any) -> Dict[str, float]:
    """Convert a dictionary‑like cell value into a Python dictionary.

    Values are expected to be of the form ``"{Key: 12.3%, Other Key: 0.0%}"``.
    Percent signs are stripped and values are converted to floats when
    possible.

    Parameters
    ----------
    cell : any
        The cell value from the sheet. May be ``NaN``, ``dict`` or ``str``.

    Returns
    -------
    dict
        Parsed dictionary with keys as strings and values as floats or strings.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}
    if isinstance(cell, dict):
        return cell
    text = str(cell).strip("{} ")
    if not text:
        return {}
    result: Dict[str, float] = {}
    for item in text.split(','):
        if not item or ':' not in item:
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
    """Construct a DataFrame from a row containing exposure dictionaries.

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
    parts: List[pd.DataFrame] = []
    for col in prefixes:
        dictionary = parse_dict(row.get(col))
        df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[col])
        parts.append(df)
    return pd.concat(parts, axis=1)


def show_performance_view() -> None:
    """Render the performance overview page using Google Sheets data, with asset class filter."""

    # Pull sheet configuration from secrets
    if not ("fund_performances" in st.secrets and "sheet_id" in st.secrets["fund_performances"]):
        st.error("Missing 'fund_performances' configuration in secrets.")
        return
    sheet_id = st.secrets["fund_performances"].get("sheet_id")
    worksheet = st.secrets["fund_performances"].get("worksheet", "Sheet1")
    df = load_sheet(sheet_id, worksheet)
    if df.empty:
        st.warning("No data returned from the performance sheet.")
        return

    # --- Load securities_master for asset class mapping ---
    if not ("securities_master" in st.secrets and "sheet_id" in st.secrets["securities_master"]):
        st.error("Missing 'securities_master' configuration in secrets.")
        return
    sec_sheet_id = st.secrets["securities_master"].get("sheet_id")
    sec_worksheet = st.secrets["securities_master"].get("worksheet", "Sheet1")
    sec_df = load_sheet(sec_sheet_id, sec_worksheet)
    if sec_df.empty or "canonical_id" not in sec_df.columns or "asset_class" not in sec_df.columns:
        st.warning("No data or missing columns in securities_master.")
        return

    # Merge asset_class into fund performances
    if "fund_id" in df.columns:
        df = df.merge(
            sec_df[["canonical_id", "asset_class"]],
            left_on="fund_id",
            right_on="canonical_id",
            how="left"
        )
    else:
        df["asset_class"] = None  # fallback if fund_id missing

    # Drop duplicates by Fund Name, Share Class, Currency and Date
    dedup_columns = [c for c in ["Fund Name", "Share Class", "Currency", "Date"] if c in df.columns]
    if dedup_columns:
        df = df.drop_duplicates(subset=dedup_columns, keep="last")

    # Filter out rows with empty "Date", "MTD", or "Fund Name" columns
    df = df[df["Date"].notna() & df["Date"].astype(str).str.strip().ne("")]
    df = df[df["MTD"].notna() & df["MTD"].astype(str).str.strip().ne("")]
    df = df[df["Fund Name"].notna() & df["Fund Name"].astype(str).str.strip().ne("")]

    # --- Filters in the same row ---
    col1, col2 = st.columns(2)
    # Asset Class Filter (do not show in table)
    asset_classes = sorted(df["asset_class"].dropna().unique().tolist())
    with col1:
        selected_asset_classes = st.multiselect(
            "Filter by Asset Class", asset_classes, default=[]
        )
    if selected_asset_classes:
        df = df[df["asset_class"].isin(selected_asset_classes)]

    # Fund selector if column exists
    with col2:
        fund_options = ["All"] + sorted(df["Fund Name"].dropna().unique().tolist()) if "Fund Name" in df.columns else []
        fund_choice = st.selectbox("Select Fund", fund_options) if fund_options else "All"
    if fund_choice != "All":
        df = df[df["Fund Name"] == fund_choice]

    # Hide unwanted columns (including any from the merged table)
    cols_to_hide = [
        "fund_id", "currency", "WTD", "YTD", "Sender", "Category", "Currency",
        "Net", "Gross", "Long Exposure", "Short Exposure", "Correct", "Received",
        "canonical_id", "asset_class"  # ensure merged columns are hidden
    ]
    df_display = df.reset_index(drop=True)
    df_display = df_display.drop(columns=[c for c in cols_to_hide if c in df_display.columns], errors="ignore")

    # Rename "Date" column to "As of date"
    if "Date" in df_display.columns:
        df_display = df_display.rename(columns={"Date": "As of date"})

    # Convert "As of date" to datetime and filter out future dates
    if "As of date" in df_display.columns:
        df_display["As of date"] = pd.to_datetime(df_display["As of date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        df_display = df_display[df_display["As of date"] <= today]

    # Reorder columns to have "Fund Name" first
    columns = df_display.columns.tolist()
    if "Fund Name" in columns:
        columns.insert(0, columns.pop(columns.index("Fund Name")))
    df_display = df_display[columns]

    # Sort by "As of date" (descending)
    if "As of date" in df_display.columns:
        df_display = df_display.sort_values("As of date", ascending=False)
        # Format date as YYYY-MM-DD
        df_display["As of date"] = df_display["As of date"].dt.strftime("%Y-%m-%d")

    st.dataframe(df_display, use_container_width=True)


def show_market_view() -> None:
    """Render the market views page using Google Sheets data with AgGrid for row selection."""

    if not ("market_views" in st.secrets and "sheet_id" in st.secrets["market_views"]):
        st.error("Missing 'market_views' configuration in secrets.")
        return
    sheet_id = st.secrets["market_views"].get("sheet_id")
    worksheet = st.secrets["market_views"].get("worksheet", "Sheet1")
    df = load_sheet(sheet_id, worksheet)
    if df.empty:
        st.warning("No data returned from the market views sheet.")
        return

    # Format the "Date" column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Define columns to hide in the main table
    cols_to_hide = [
        "Document Type", "Data & Evidence", "Key Themes", "Risks/Uncertainties",
        "Risks / Uncertainties", "Evidence Strength & Uniqueness", "Evidence Strenght & Uniqueness",
        "Follow-up Actions", "Title"
    ]
    # Prepare main table
    df_display = df.drop(columns=[c for c in cols_to_hide if c in df.columns], errors="ignore")

    # Reorder columns as requested
    desired_order = [
        "Asset Class & Region", "Date", "Author(s)", "Institution/Source",
        "Market Regime/Context", "Instrument Name"
    ]
    columns = [col for col in desired_order if col in df_display.columns]
    columns += [col for col in df_display.columns if col not in columns]
    df_display = df_display[columns]

    # Column filter interface
    filter_columns = st.multiselect("Columns to filter", df_display.columns.tolist())
    filtered = df_display.copy()
    for col in filter_columns:
        choices = sorted(df_display[col].dropna().unique().tolist())
        selected = st.multiselect(f"{col}", choices)
        if selected:
            filtered = filtered[filtered[col].isin(selected)]

    # AGGrid for interactive table and row selection
    gb = GridOptionsBuilder.from_dataframe(filtered)
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()
    grid_response = AgGrid(
        filtered,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=False,
        theme="streamlit",
        update_mode="SELECTION_CHANGED",
        fit_columns_on_grid_load=True
    )
    selected = grid_response['selected_rows']

    # Robust row matching for details
    def get_full_row(row):
        # Try to match on all available columns used in the table
        mask = pd.Series([True] * len(df))
        for key in ["Date", "Asset Class & Region", "Institution/Source"]:
            if key in df.columns and key in row:
                mask &= (df[key] == row.get(key))
        matches = df[mask]
        if not matches.empty:
            return matches.iloc[0]
        return row  # fallback

    if isinstance(selected, list) and len(selected) > 0:
        row = selected[0]
        full_row = get_full_row(row)
        with st.expander("Market View Details", expanded=True):
            all_columns = list(filtered.columns) + [c for c in cols_to_hide if c in df.columns]
            cols = st.columns(2)
            for i, col in enumerate(all_columns):
                with cols[i % 2]:
                    st.markdown(f"**{col}:**")
                    value = full_row[col] if col in full_row else row.get(col)
                    st.markdown(value if pd.notna(value) else "_(empty)_")
    elif hasattr(selected, "empty") and not selected.empty:
        row = selected.iloc[0]
        full_row = get_full_row(row)
        with st.expander("Market View Details", expanded=True):
            all_columns = list(filtered.columns) + [c for c in cols_to_hide if c in df.columns]
            cols = st.columns(2)
            for i, col in enumerate(all_columns):
                with cols[i % 2]:
                    st.markdown(f"**{col}:**")
                    value = full_row[col] if col in full_row else row.get(col)
                    st.markdown(value if pd.notna(value) else "_(empty)_")

def show_fund_monitor() -> None:
    """Render the fund monitor page using Google Sheets data, using canonical fund list and improved UX."""

    # Load exposures sheet
    if not ("exposures" in st.secrets and "sheet_id" in st.secrets["exposures"]):
        st.error("Missing 'exposures' configuration in secrets.")
        return
    sheet_id = st.secrets["exposures"].get("sheet_id")
    worksheet = st.secrets["exposures"].get("worksheet", "Sheet1")
    df = load_sheet(sheet_id, worksheet)
    if df.empty:
        st.warning("No data returned from the exposures sheet.")
        return

    # Load securities_master for canonical fund list
    if not ("securities_master" in st.secrets and "sheet_id" in st.secrets["securities_master"]):
        st.error("Missing 'securities_master' configuration in secrets.")
        return
    sec_sheet_id = st.secrets["securities_master"].get("sheet_id")
    sec_worksheet = st.secrets["securities_master"].get("worksheet", "Sheet1")
    sec_df = load_sheet(sec_sheet_id, sec_worksheet)
    if sec_df.empty or "canonical_id" not in sec_df.columns or "canonical_name" not in sec_df.columns:
        st.warning("No data or missing columns in securities_master.")
        return

    # Only include funds present in exposures
    exposure_fund_ids = set(df["fund_id"].dropna().unique()) if "fund_id" in df.columns else set()
    canonical_funds = sec_df[sec_df["canonical_id"].isin(exposure_fund_ids)][["canonical_id", "canonical_name"]].drop_duplicates()
    canonical_funds = canonical_funds.sort_values("canonical_name")
    fund_options = canonical_funds["canonical_name"].tolist()
    default_fund = None
    if "defaults" in st.secrets and "fund" in st.secrets["defaults"]:
        default_fund = st.secrets["defaults"]["fund"]
    fund_index = fund_options.index(default_fund) if default_fund in fund_options else 0
    fund_choice = st.selectbox("Select Fund", fund_options, index=fund_index)

    # Map selected canonical_name to canonical_id
    selected_canonical_id = canonical_funds[canonical_funds["canonical_name"] == fund_choice]["canonical_id"].iloc[0]

    # Format the date column homogenously
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Filter exposures by canonical_id (assuming exposures has a 'fund_id' column)
    if "fund_id" in df.columns:
        fund_df = df[df["fund_id"] == selected_canonical_id]
    else:
        st.error("No fund_id column in exposures sheet.")
        return

    # Remove duplicate dates for selection
    if fund_df.empty:
        st.warning("No records found for the selected fund.")
        return
    date_values = sorted(fund_df["date"].dropna().unique().tolist(), reverse=True)
    date_choice = st.selectbox("Select Date", date_values)

    # File type selection
    file_type = st.selectbox("Select file type", ["pdf", "img"], index=0)

    # Filter by date and file_type (if column exists)
    filtered_row = fund_df[(fund_df["date"] == date_choice)]
    if "file_type" in filtered_row.columns:
        filtered_row = filtered_row[filtered_row["file_type"] == file_type]
    if filtered_row.empty:
        st.warning("No records found for the selected date and file type.")
        return
    row = filtered_row.iloc[0]

    # Display metrics
    metrics_cols = st.columns(5)
    metrics_cols[0].metric("AUM", row.get("aum_fund") or row.get("aum_firm"))
    metrics_cols[1].metric("Net", row.get("net"))
    metrics_cols[2].metric("Gross", row.get("gross"))
    metrics_cols[3].metric("Long", row.get("long"))
    metrics_cols[4].metric("Short", row.get("short"))

    # Positions counts
    st.subheader("Number of Positions")
    pos_cols = st.columns(2)
    pos_cols[0].metric("Long", row.get("num_pos_long"))
    pos_cols[1].metric("Short", row.get("num_pos_short"))

    # Sector and Geographic exposures in the same row
    sector_keys = ["sector_long", "sector_short", "sector_gross", "sector_net"]
    geo_keys = ["geo_long", "geo_short", "geo_gross", "geo_net"]
    st.subheader("Exposures")
    exp_cols = st.columns(2)
    if all(k in row.index for k in sector_keys):
        with exp_cols[0]:
            st.markdown("**Sector Exposures**")
            sector_df = build_exposure_df(row, sector_keys)
            st.dataframe(sector_df)
    if all(k in row.index for k in geo_keys):
        with exp_cols[1]:
            st.markdown("**Geographical Exposures**")
            geo_df = build_exposure_df(row, geo_keys)
            st.dataframe(geo_df)

    # --- Historical Analysis Section ---
    st.subheader("Historical Analysis")
    # Filter for selected fund and remove duplicate dates
    hist_df = fund_df.copy()
    hist_df = hist_df.drop_duplicates(subset=["date"])
    # Only plot if net and gross columns exist and are numeric
    if "net" in hist_df.columns and "gross" in hist_df.columns:
        # Convert to numeric, coerce errors
        hist_df["net"] = pd.to_numeric(hist_df["net"], errors="coerce")
        hist_df["gross"] = pd.to_numeric(hist_df["gross"], errors="coerce")
        hist_df = hist_df.dropna(subset=["date", "net", "gross"])
        hist_df = hist_df.sort_values("date")
        if not hist_df.empty:
            import altair as alt
            chart = alt.Chart(hist_df).transform_fold(
                ["net", "gross"],
                as_=["Exposure", "Value"]
            ).mark_line(point=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Value:Q", title="Exposure"),
                color=alt.Color("Exposure:N", title="Type"),
                tooltip=["date", "Exposure", "Value"]
            ).properties(width=700, height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No historical net/gross data available for this fund.")
    else:
        st.info("No historical net/gross data available for this fund.")

    
def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Fund Monitoring Dashboard", layout="wide")

    # Sidebar navigation using option_menu
    with st.sidebar:
        page = option_menu(
            "Navigation",
            ["Performance Est", "Market Views", "Fund Monitor"],
            # icons=["bar-chart", "globe", "clipboard-data"],  # optional: choose icons
            # menu_icon="cast",  # optional: sidebar header icon
            default_index=0,
            orientation="vertical"
        )

    if page == "Performance Est":
        # st.title("Fund Monitoring Dashboard")
        st.header("Performance Estimates")
        show_performance_view()
    elif page == "Market Views":
        # st.title("Fund Monitoring Dashboard")
        st.header("Market Views")
        show_market_view()
    elif page == "Fund Monitor":
        # st.title("Fund Monitoring Dashboard")
        st.header("Fund Monitor")
        show_fund_monitor()


if __name__ == "__main__":
    main()