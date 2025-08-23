# app_gsheets.py
"""
Streamlit application for monitoring investment funds using data
directly from Google Sheets.

Adds "Market Analytics" with:
- Monthly Seasonality Explorer
- Market Memory Explorer
- Breakout Scanner
- 10-Year Nominal & Real Yield Dashboard
- Liquidity & Fed Policy Tracker
- Market Stress Composite

Existing pages retained: Performance Est, Market Views, Fund Monitor.
"""

import os
from typing import Dict, List, Tuple
import math
import json
import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import gspread
from st_aggrid import AgGrid, GridOptionsBuilder

# New: markets/fetching libs
import yfinance as yf
from pandas_datareader import data as pdr

# Google auth pieces (existing)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import matplotlib.pyplot as plt

# near other imports
from typing import Any
try:
    from pandas.io.formats.style import Styler
except Exception:  # pandas build without exposed Styler path
    Styler = Any  # type: ignore


# --- Drive scopes / helpers (existing) ---
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _drive_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=DRIVE_SCOPES)
    return build('drive', 'v3', credentials=creds)

def _download_json_from_drive(file_id: str):
    svc = _drive_client()
    try:
        meta = svc.files().get(
            fileId=file_id,
            fields="id,name,mimeType",
            supportsAllDrives=True
        ).execute()
        mime = meta["mimeType"]

        if mime.startswith("application/vnd.google-apps"):
            resp = svc.files().export(fileId=file_id, mimeType="text/plain").execute()
            raw = resp if isinstance(resp, bytes) else resp.encode("utf-8", errors="ignore")
        else:
            from io import BytesIO
            from googleapiclient.http import MediaIoBaseDownload
            req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            raw = fh.getvalue()

        text = raw.decode("utf-8", errors="ignore").replace("\ufeff", "").strip()
        first_brace = text.find("{")
        first_bracket = text.find("[")
        candidates = [i for i in (first_brace, first_bracket) if i != -1]
        if candidates:
            start = min(candidates)
            if start > 0:
                text = text[start:]
        if not text or text[0] not in "{[":
            raise ValueError("Empty or non‑JSON content from Drive file")
        return json.loads(text)

    except (HttpError, ValueError, json.JSONDecodeError) as e:
        st.error(f"Failed to parse track_record.json: {e}")
        return None

def _normalize_track_record(obj):
    if obj is None:
        return None
    if isinstance(obj, dict) and "returns" in obj and isinstance(obj["returns"], list):
        return {"returns": obj["returns"]}
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return {"returns": obj["data"]}
    if isinstance(obj, list) and obj:
        for item in obj:
            if isinstance(item, dict) and isinstance(item.get("data"), list):
                return {"returns": item["data"], "meta": item.get("meta"), "track_label": item.get("track_label")}
        if all(isinstance(x, dict) and {"date","return"} <= set(x.keys()) for x in obj):
            return {"returns": obj}
    st.warning("track_record.json structure not recognized; expected a list/dict with 'data' or 'returns'.")
    return None

def fetch_track_record_json(fund_id: str) -> dict | None:
    svc = _drive_client()
    parent_folder_id = st.secrets["drive"]["parent_folder_id"]
    q_folder = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{fund_id}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    resp = svc.files().list(
        q=q_folder, fields="files(id,name)", includeItemsFromAllDrives=True,
        supportsAllDrives=True, corpora="allDrives"
    ).execute()
    folders = resp.get("files", [])
    if not folders:
        st.warning(f"No Google Drive folder found for fund_id: {fund_id} in parent folder.")
        return None
    folder_id = folders[0]["id"]

    q_file = f"'{folder_id}' in parents and name = 'track_record.json' and trashed = false"
    resp = svc.files().list(
        q=q_file, fields="files(id,name,mimeType)", includeItemsFromAllDrives=True,
        supportsAllDrives=True, corpora="allDrives"
    ).execute()
    files = resp.get("files", [])
    if not files:
        st.warning(f"No track_record.json found in folder for fund_id: {fund_id}")
        return None
    obj = _download_json_from_drive(files[0]["id"])
    return _normalize_track_record(obj)

# --- Sheets auth/helpers (existing) ---
def get_gspread_client() -> gspread.Client:
    # Mirrors existing credential resolution:contentReference[oaicite:2]{index=2}
    if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        try:
            client = gspread.service_account_from_dict(creds_dict)
            return client
        except Exception as exc:
            st.error(f"Failed to authenticate using gcp_service_account in secrets: {exc}")
            raise
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and os.path.exists(creds_path):
        try:
            client = gspread.service_account(filename=creds_path)
            return client
        except Exception as exc:
            st.error(f"Failed to authenticate using GOOGLE_APPLICATION_CREDENTIALS at {creds_path}: {exc}")
            raise
    raise RuntimeError(
        "Google Sheets credentials not found. Provide a service account key via "
        "st.secrets['gcp_service_account'] or set the GOOGLE_APPLICATION_CREDENTIALS environment variable."
    )

def load_sheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
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
    parts: List[pd.DataFrame] = []
    for col in prefixes:
        dictionary = parse_dict(row.get(col))
        df = pd.DataFrame.from_dict(dictionary, orient='index', columns=[col])
        parts.append(df)
    return pd.concat(parts, axis=1)


# === New page: Fund Database ===  # :contentReference[oaicite:0]{index=0}
def show_fund_database() -> None:
    import pandas as pd
    import streamlit as st
    from st_aggrid import AgGrid, GridOptionsBuilder

    tabs = st.tabs(["Overview", "Liquidity & Ops"])

    # --- Tab 1: Overview (editable, non-destructive save) ---
    with tabs[0]:
        if not ("fund_database" in st.secrets and "sheet_id" in st.secrets["fund_database"]):
            st.error("Missing 'fund_database' configuration in secrets.")
            return
        sheet_id = st.secrets["fund_database"]["sheet_id"]
        worksheet = st.secrets["fund_database"].get("worksheet", "fund database")

        df = load_sheet(sheet_id, worksheet)
        if df.empty:
            st.warning("No rows found in the 'fund database' sheet.")
            df = pd.DataFrame()

        f1, f2, f3 = st.columns(3)
        with f1:
            macro_vals = sorted(df["Brightside Macro"].dropna().unique().tolist()) if "Brightside Macro" in df.columns else []
            sel_macro = st.multiselect("Brightside Macro", macro_vals, default=[], key="fd_ov_macro")
        with f2:
            status_vals = sorted(df["Status"].dropna().unique().tolist()) if "Status" in df.columns else []
            sel_status = st.multiselect("Status", status_vals, default=[], key="fd_ov_status")
        with f3:
            asset_vals = sorted(df["Asset Class"].dropna().unique().tolist()) if "Asset Class" in df.columns else []
            sel_asset = st.multiselect("Asset Class", asset_vals, default=[], key="fd_ov_asset")

        filtered = df.copy()
        if sel_macro and "Brightside Macro" in filtered.columns:
            filtered = filtered[filtered["Brightside Macro"].isin(sel_macro)]
        if sel_status and "Status" in filtered.columns:
            filtered = filtered[filtered["Status"].isin(sel_status)]
        if sel_asset and "Asset Class" in filtered.columns:
            filtered = filtered[filtered["Asset Class"].isin(sel_asset)]

        _ALLOWED_COLS = [
            "Fund Name",
            "Manager",
            "Asset Class",
            "Type",
            "Management Fee",
            "Performance Fee",
            "Inception",
            "AUM (in USD Millions)",
        ]
        show_cols = [c for c in _ALLOWED_COLS if c in filtered.columns]
        display_df = filtered[show_cols].copy() if show_cols else filtered.copy()
        display_df = display_df.reindex(columns=show_cols)

        # top_l, top_r = st.columns([1, 0.18])
        # with top_r:
        #     # do_save = st.button("save changes", type="primary", use_container_width=True, key="fd_ov_save")

        gb = GridOptionsBuilder.from_dataframe(display_df if not display_df.empty else pd.DataFrame(columns=_ALLOWED_COLS))
        gb.configure_default_column(editable=True, resizable=True, filter=True)
        gb.configure_column("Fund Name", editable=False)
        gb.configure_grid_options(rowSelection="single")
        grid = AgGrid(
            display_df,
            gridOptions=gb.build(),
            theme="streamlit",
            enable_enterprise_modules=False,
            allow_unsafe_jscode=False,
            fit_columns_on_grid_load=True,
            update_mode="MODEL_CHANGED",
            reload_data=False,
            key="fd_ov_grid",
        )
        edited_df = pd.DataFrame(grid["data"])

        # if do_save:
        #     def _normalize(cell):
        #         if cell is None:
        #             return ""
        #         return str(cell).strip()

        #     def _build_key(row_dict: dict) -> str:
        #         return f"{_normalize(row_dict.get('Fund Name'))}||{_normalize(row_dict.get('Manager'))}"

        #     def _diff_and_update_sheet(edited_df_: pd.DataFrame, sheet_id_: str, worksheet_name_: str) -> tuple[int, int]:
        #         from gspread.models import Cell

        #         client = get_gspread_client()
        #         sh = client.open_by_key(sheet_id_)
        #         ws = sh.worksheet(worksheet_name_)

        #         values = ws.get_all_values()
        #         if not values:
        #             raise RuntimeError("Target worksheet is empty; cannot map columns/rows.")

        #         header = values[0]
        #         col_idx = {name: (header.index(name) + 1) for name in header if name}
        #         missing = [c for c in _ALLOWED_COLS if c not in col_idx]
        #         if missing:
        #             raise RuntimeError(f"Missing columns in sheet: {missing}")

        #         key_to_rownum = {}
        #         for i, row_vals in enumerate(values[1:], start=2):
        #             row_dict = {h: (row_vals[j] if j < len(row_vals) else "") for j, h in enumerate(header)}
        #             key = _build_key(row_dict)
        #             if key:
        #                 key_to_rownum[key] = i

        #         keep_cols = [c for c in _ALLOWED_COLS if c in edited_df_.columns]
        #         edited_view = edited_df_[keep_cols].copy()
        #         edited_view["_key"] = edited_view.apply(lambda s: _build_key(s.to_dict()), axis=1)

        #         cells = []
        #         rows_touched = set()

        #         for _, row in edited_view.iterrows():
        #             key = row["_key"]
        #             if not key or key not in key_to_rownum:
        #                 continue
        #             rnum = key_to_rownum[key]
        #             rows_touched.add(rnum)
        #             raw = values[rnum - 1] if rnum - 1 < len(values) else []
        #             sheet_row_dict = {h: (raw[i] if i < len(raw) else "") for i, h in enumerate(header)}

        #             for col in keep_cols:
        #                 new_val = _normalize(row.get(col))
        #                 old_val = _normalize(sheet_row_dict.get(col))
        #                 if new_val != old_val:
        #                     cells.append(Cell(row=rnum, col=col_idx[col], value=new_val))

        #         if cells:
        #             ws.update_cells(cells, value_input_option="USER_ENTERED")
        #         return (len(cells), len(rows_touched))

        #     try:
        #         changed, touched = _diff_and_update_sheet(edited_df, sheet_id, worksheet)
        #         st.success(f"Patched {changed} cells across {touched} rows.")
        #     except Exception as exc:
        #         st.error(f"Failed to save to Google Sheet: {exc}")

    # --- Tab 2: Liquidity & Ops (read-only grid with same filters) ---
    with tabs[1]:
        if not ("fund_database" in st.secrets and "sheet_id" in st.secrets["fund_database"]):
            st.error("Missing 'fund_database' configuration in secrets.")
            return
        sheet_id = st.secrets["fund_database"]["sheet_id"]
        worksheet = st.secrets["fund_database"].get("worksheet", "fund database")

        df = load_sheet(sheet_id, worksheet)
        if df.empty:
            st.warning("No rows found in the 'fund database' sheet.")
            df = pd.DataFrame()

        f1, f2, f3 = st.columns(3)
        with f1:
            macro_vals = sorted(df["Brightside Macro"].dropna().unique().tolist()) if "Brightside Macro" in df.columns else []
            sel_macro = st.multiselect("Brightside Macro", macro_vals, default=[], key="fd_lo_macro")
        with f2:
            status_vals = sorted(df["Status"].dropna().unique().tolist()) if "Status" in df.columns else []
            sel_status = st.multiselect("Status", status_vals, default=[], key="fd_lo_status")
        with f3:
            asset_vals = sorted(df["Asset Class"].dropna().unique().tolist()) if "Asset Class" in df.columns else []
            sel_asset = st.multiselect("Asset Class", asset_vals, default=[], key="fd_lo_asset")

        filtered = df.copy()
        if sel_macro and "Brightside Macro" in filtered.columns:
            filtered = filtered[filtered["Brightside Macro"].isin(sel_macro)]
        if sel_status and "Status" in filtered.columns:
            filtered = filtered[filtered["Status"].isin(sel_status)]
        if sel_asset and "Asset Class" in filtered.columns:
            filtered = filtered[filtered["Asset Class"].isin(sel_asset)]

        liquidity_cols = [
            "Fund Name",
            "Redemption Term",
            "Notice days before redemption",
            "Lock Up",
            "Level Gate",
            "Fund Administrator",
            "Fund Domicile",
            "Firm Domicile",
        ]
        show_cols2 = [c for c in liquidity_cols if c in filtered.columns]
        display_df2 = filtered[show_cols2].copy() if show_cols2 else filtered.copy()
        display_df2 = display_df2.reindex(columns=show_cols2)

        gb2 = GridOptionsBuilder.from_dataframe(display_df2 if not display_df2.empty else pd.DataFrame(columns=liquidity_cols))
        gb2.configure_default_column(editable=False, resizable=True, filter=True)
        gb2.configure_grid_options(rowSelection="single")
        AgGrid(
            display_df2,
            gridOptions=gb2.build(),
            theme="streamlit",
            enable_enterprise_modules=False,
            allow_unsafe_jscode=False,
            fit_columns_on_grid_load=True,
            update_mode="NO_UPDATE",
            reload_data=False,
            key="fd_lo_grid",
        )


# ---- Existing product pages (kept as-is) ----

def show_performance_view() -> None:
    # Uses fund_performances and securities_master like your original implementation:contentReference[oaicite:3]{index=3}
    if not ("fund_performances" in st.secrets and "sheet_id" in st.secrets["fund_performances"]):
        st.error("Missing 'fund_performances' configuration in secrets.")
        return
    sheet_id = st.secrets["fund_performances"].get("sheet_id")
    worksheet = st.secrets["fund_performances"].get("worksheet", "Sheet1")
    df = load_sheet(sheet_id, worksheet)
    if df.empty:
        st.warning("No data returned from the performance sheet.")
        return

    if not ("securities_master" in st.secrets and "sheet_id" in st.secrets["securities_master"]):
        st.error("Missing 'securities_master' configuration in secrets.")
        return
    sec_sheet_id = st.secrets["securities_master"].get("sheet_id")
    sec_worksheet = st.secrets["securities_master"].get("worksheet", "Sheet1")
    sec_df = load_sheet(sec_sheet_id, sec_worksheet)
    if sec_df.empty or "canonical_id" not in sec_df.columns or "asset_class" not in sec_df.columns:
        st.warning("No data or missing columns in securities_master.")
        return

    if "fund_id" in df.columns:
        df = df.merge(
            sec_df[["canonical_id", "asset_class"]],
            left_on="fund_id", right_on="canonical_id", how="left"
        )
    else:
        df["asset_class"] = None

    dedup_columns = [c for c in ["Fund Name", "Share Class", "Currency", "Date"] if c in df.columns]
    if dedup_columns:
        df = df.drop_duplicates(subset=dedup_columns, keep="last")

    df = df[df["Date"].notna() & df["Date"].astype(str).str.strip().ne("")]
    df = df[df["MTD"].notna() & df["MTD"].astype(str).str.strip().ne("")]
    df = df[df["Fund Name"].notna() & df["Fund Name"].astype(str).str.strip().ne("")]

    col1, col2 = st.columns(2)
    asset_classes = sorted(df["asset_class"].dropna().unique().tolist())
    with col1:
        selected_asset_classes = st.multiselect("Filter by Asset Class", asset_classes, default=[])
    if selected_asset_classes:
        df = df[df["asset_class"].isin(selected_asset_classes)]
    with col2:
        fund_options = ["All"] + sorted(df["Fund Name"].dropna().unique().tolist()) if "Fund Name" in df.columns else []
        fund_choice = st.selectbox("Select Fund", fund_options) if fund_options else "All"
    if fund_choice != "All":
        df = df[df["Fund Name"] == fund_choice]

    cols_to_hide = [
        "fund_id","currency","WTD","YTD","Sender","Category","Currency","Net","Gross",
        "Long Exposure","Short Exposure","Correct","canonical_id","asset_class" # "Received"
    ]
    df_display = df.reset_index(drop=True)
    df_display = df_display.drop(columns=[c for c in cols_to_hide if c in df_display.columns], errors="ignore")
    if "Date" in df_display.columns:
        df_display = df_display.rename(columns={"Date": "As of date"})
        df_display["As of date"] = pd.to_datetime(df_display["As of date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        df_display = df_display[df_display["As of date"] <= today]
    columns = df_display.columns.tolist()
    if "Fund Name" in columns:
        columns.insert(0, columns.pop(columns.index("Fund Name")))
    df_display = df_display[columns]
    if "As of date" in df_display.columns:
        df_display = df_display.sort_values("As of date", ascending=False)
        df_display["As of date"] = df_display["As of date"].dt.strftime("%Y-%m-%d")
    if "MTD" in df_display.columns:
        df_display["MTD"] = df_display["MTD"].astype(str).fillna("")
    if "Received" in df_display.columns:
        df_display["Received"] = pd.to_datetime(df_display["Received"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ======== NEW HELPERS FOR "MARKET VIEWS" ========

@st.cache_data(show_spinner=False, ttl=3600)
def _load_letters() -> pd.DataFrame:
    """
    Load 'fund letters' from Google Sheets as configured in st.secrets["fund_letters"].
    Expected keys: sheet_id, worksheet. Returns a DataFrame (may be empty).
    """
    if not ("letters" in st.secrets and "sheet_id" in st.secrets["letters"]):
        st.error("Missing 'letters' configuration in secrets.")
        return pd.DataFrame()
    sheet_id = st.secrets["letters"]["sheet_id"]
    worksheet = st.secrets["letters"].get("worksheet", "Sheet1")
    df = load_sheet(sheet_id, worksheet)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def _load_fund_master() -> pd.DataFrame:
    """
    Load securities master to map fund_id -> fund name.
    Expects columns: canonical_id, canonical_name.
    """
    if not ("securities_master" in st.secrets and "sheet_id" in st.secrets["securities_master"]):
        st.error("Missing 'securities_master' configuration in secrets.")
        return pd.DataFrame()
    sec_sheet_id = st.secrets["securities_master"]["sheet_id"]
    sec_worksheet = st.secrets["securities_master"].get("worksheet", "Sheet1")
    sec_df = load_sheet(sec_sheet_id, sec_worksheet)
    needed = {"canonical_id", "canonical_name"}
    if sec_df.empty or not needed.issubset(sec_df.columns):
        st.warning("securities_master missing required columns: canonical_id, canonical_name")
        return pd.DataFrame()
    return sec_df[["canonical_id", "canonical_name"]].drop_duplicates()

def _map_fund_names(df: pd.DataFrame, fund_id_col: str = "fund_id") -> pd.DataFrame:
    """
    Append 'fund_name' by joining letters.df[fund_id] -> securities_master.canonical_name
    """
    master = _load_fund_master()
    if df.empty or master.empty or fund_id_col not in df.columns:
        df = df.copy()
        df["fund_name"] = df.get(fund_id_col, "")
        return df
    out = df.merge(master, left_on=fund_id_col, right_on="canonical_id", how="left")
    out["fund_name"] = out["canonical_name"].fillna(out[fund_id_col].astype(str))
    return out.drop(columns=["canonical_id", "canonical_name"], errors="ignore")

def _nonnull_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().ne("") & s.notna()

def _parse_report_date(df: pd.DataFrame, col: str = "report_date") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

CRYPTO_TICKERS = {
    "BNB","PHA","TON","JUP","JTO","UNI","HNT","PYTH","DYDX",
    "BTC","ETH","SOL","ADA","AVAX","DOGE","LINK"
}
_SPECIAL_MAP = {
    "EUR": "EURUSD=X",   # FX
    "BRENT": "BZ=F",     # ICE Brent futures
}

def _resolve_yf_symbol(t: str | None) -> str | None:
    if t is None:
        return None
    u = str(t).strip().upper()
    if u in _SPECIAL_MAP:
        return _SPECIAL_MAP[u]
    if u.endswith("-USD") or u.endswith("=X") or "." in u:
        return u
    if u in CRYPTO_TICKERS:
        return f"{u}-USD"
    if len(u) == 3 and u.isalpha():   # generic FX like "EUR"
        return f"{u}USD=X"
    return u

@st.cache_data(show_spinner=False, ttl=900)
def _yahoo_history_panel(tickers: list[str], lookback_days: int = 750) -> pd.DataFrame:
    """
    Robust per-symbol downloader. Works even when bulk download drops all data.
    Returns tidy frame: [date, ticker, adj_close], with 'ticker' equal to the ORIGINAL symbol.
    """
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    # map original -> yahoo symbol
    originals = [t for t in tickers if isinstance(t, str) and t.strip()]
    mapping = {t.upper(): _resolve_yf_symbol(t) for t in originals}

    frames = []
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)

    for orig_sym, yf_sym in mapping.items():
        if not yf_sym:
            continue

        df_sym = pd.DataFrame()
        # Attempt #1: standard download with explicit date range
        try:
            df_sym = yf.download(
                tickers=yf_sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
                raise_errors=False,
            )
        except Exception:
            df_sym = pd.DataFrame()

        # Attempt #2: per-ticker history if #1 empty or all NaN
        if df_sym is None or df_sym.empty or (
            isinstance(df_sym.columns, pd.Index) and df_sym.dropna(how="all").empty
        ):
            try:
                h = yf.Ticker(yf_sym).history(period="730d", interval="1d", auto_adjust=True)
                df_sym = h
            except Exception:
                df_sym = pd.DataFrame()

        # Normalize to tidy: prefer Adj Close, fallback to Close
        col = None
        if isinstance(df_sym.columns, pd.MultiIndex):
            if (yf_sym, "Adj Close") in df_sym.columns:
                col = ("Adj Close", True)
            elif (yf_sym, "Close") in df_sym.columns:
                col = ("Close", True)
        else:
            if "Adj Close" in df_sym.columns:
                col = ("Adj Close", False)
            elif "Close" in df_sym.columns:
                col = ("Close", False)

        if not col:
            continue

        if col[1]:  # from multiindex
            series = df_sym[(yf_sym, col[0])].dropna()
        else:
            series = df_sym[col[0]].dropna()

        if series.empty:
            continue

        sub = series.to_frame("adj_close").reset_index()
        # unify date column name for both code paths
        if "Date" in sub.columns:
            sub = sub.rename(columns={"Date": "date"})
        elif "index" in sub.columns:
            sub = sub.rename(columns={"index": "date"})

        sub["date"] = pd.to_datetime(sub["date"], errors="coerce", utc=True).dt.tz_localize(None)
        sub = sub[(sub["date"] >= start) & (sub["date"] <= end)]
        if sub.empty:
            continue

        sub["ticker"] = orig_sym  # keep original symbol
        frames.append(sub[["date", "ticker", "adj_close"]])

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "adj_close"]).sort_values(["ticker", "date"])
    return out


def _return_slice(px: pd.DataFrame, when: str) -> pd.Series:
    """
    Compute percentage changes per ticker for the given horizon.
    when in {"1d","1w","mtd","ytd"}.
    """
    if px.empty:
        return pd.Series(dtype=float)
    today = pd.Timestamp.today().normalize()
    # last valid close per ticker
    last_px = px.groupby("ticker")["adj_close"].last()
    if when == "1d":
        prev = px.groupby("ticker")["adj_close"].nth(-2)
        return ((last_px / prev) - 1.0) * 100.0
    if when == "1w":
        # 5 trading days ~ 7 calendar days; nth(-6) guards short series
        prev = px.groupby("ticker")["adj_close"].nth(-6)
        return ((last_px / prev) - 1.0) * 100.0
    if when == "mtd":
        month_start = today.replace(day=1)
        ref = px[px["date"] <= today].copy()
        ref = ref.merge(
            ref.groupby("ticker")["date"].apply(lambda s: s[s >= month_start].min()).rename("anchor"),
            on="ticker", how="left"
        )
        ref = ref[ref["date"] == ref["anchor"]]
        ref_px = ref.set_index("ticker")["adj_close"]
        return ((last_px / ref_px) - 1.0) * 100.0
    if when == "ytd":
        year_start = today.replace(month=1, day=1)
        ref = px[px["date"] <= today].copy()
        ref = ref.merge(
            ref.groupby("ticker")["date"].apply(lambda s: s[s >= year_start].min()).rename("anchor"),
            on="ticker", how="left"
        )
        ref = ref[ref["date"] == ref["anchor"]]
        ref_px = ref.set_index("ticker")["adj_close"]
        return ((last_px / ref_px) - 1.0) * 100.0
    return pd.Series(dtype=float)

def _attach_return_columns(df: pd.DataFrame, ticker_col: str = "position_ticker") -> pd.DataFrame:
    # Only MTD% and YTD%
    if df.empty or ticker_col not in df.columns:
        for c in ["MTD %","YTD %"]:
            df[c] = None
        return df

    tickers = [t for t in df[ticker_col].astype(str).str.upper().tolist() if t and t != "NAN"]
    px = _yahoo_history_panel(tickers)
    if px.empty:
        for c in ["MTD %","YTD %"]:
            df[c] = None
        return df

    rmtd = _return_slice(px, "mtd")
    rytd = _return_slice(px, "ytd")

    out = df.copy()

    def _map(series: pd.Series, t):
        v = series.get(str(t).upper())
        if v is None or pd.isna(v):
            return None
        try:
            return round(float(v), 2)
        except Exception:
            return None

    out["MTD %"] = out[ticker_col].apply(lambda t: _map(rmtd, t))
    out["YTD %"] = out[ticker_col].apply(lambda t: _map(rytd, t))
    return out


def _attach_since_report_col(
    df: pd.DataFrame,
    ticker_col: str = "position_ticker",
    report_date_col: str = "report_date",
    colname: str = "Since Report %",
) -> pd.DataFrame:
    """
    Adds a column with % change from the first available close on/after report_date to latest close.
    Uses _yahoo_history_panel for prices. Returns df with the new column.
    """
    out = df.copy()
    out[colname] = None

    if out.empty or ticker_col not in out.columns or report_date_col not in out.columns:
        return out

    # Ensure datetimes are tz-naive to match _yahoo_history_panel filtering
    out[report_date_col] = pd.to_datetime(out[report_date_col], errors="coerce")
    tickers = (
        out[ticker_col].dropna().astype(str).str.upper().str.strip()
        .replace({"NAN": None})
        .dropna().unique().tolist()
    )
    if not tickers:
        return out

    # Lookback = from earliest report_date to today, with small buffer; clamp 30..1500
    today = pd.Timestamp.today().normalize()
    min_rep = out[report_date_col].dropna().min()
    if pd.isna(min_rep):
        lookback_days = 365
    else:
        lookback_days = int((today - min_rep).days) + 10
        lookback_days = max(30, min(lookback_days, 1500))

    px = _yahoo_history_panel(tickers, lookback_days=lookback_days)
    if px.empty:
        return out

    # Precompute latest close per ticker once
    last_close = px.sort_values(["ticker", "date"]).groupby("ticker")["adj_close"].last()

    # Row-wise compute anchor and return
    def _since_row(tkr, rd):
        if pd.isna(rd) or not isinstance(tkr, str) or not tkr.strip():
            return None
        t = tkr.upper().strip()
        if t not in last_close.index:
            return None
        df_t = px[px["ticker"] == t]
        # first close on/after report date
        anchor = df_t.loc[df_t["date"] >= rd]
        if anchor.empty:
            return None
        a = float(anchor.iloc[0]["adj_close"])
        z = float(last_close.loc[t])
        if a <= 0:
            return None
        return round((z / a - 1.0) * 100.0, 2)

    out[colname] = [
        _since_row(t, d) for t, d in zip(out[ticker_col], out[report_date_col])
    ]
    return out


# ======== REPLACE show_market_view WITH THIS ========

def show_market_view() -> None:
    letters = _load_letters()
    if letters.empty:
        st.warning("Letters table is empty or unavailable.")
        return
    letters = _parse_report_date(letters, "report_date")
    letters = _map_fund_names(letters, fund_id_col="fund_id")

    # ── Row 1 ─────────────────────────────────────────────────────────────
    top_tabs = st.tabs([
        "Fund Positions",
        "Investment Thesis",
    ])

    with top_tabs[0]:
        df = letters.copy()
        for col in ["position_ticker","position_weight_percent"]:
            if col not in df.columns:
                df[col] = None
        mask = _nonnull_mask(df["position_ticker"]) & _nonnull_mask(df["position_weight_percent"])
        df = df[mask].copy()

        if df.empty:
            st.info("No rows with non-empty position_ticker and position_weight_percent.")
        else:
            df = _parse_report_date(df, "report_date")
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            latest = df.groupby("fund_name")["report_date"].transform("max")
            df = df[df["report_date"] == latest].copy()

            for col in ["position_name","position_sector"]:
                if col not in df.columns:
                    df[col] = None

            view = df[[
                "fund_name","position_name","position_ticker","position_sector","position_weight_percent","report_date"
            ]].rename(columns={
                "fund_name":"Fund Name",
                "position_name":"Position Name",
                "position_ticker":"Position Ticker",
                "position_sector":"Position Sector",
                "position_weight_percent":"Position Weight (%)",
                "report_date":"Report Date"
            }).sort_values(["Fund Name","Position Weight (%)"], ascending=[True, False])

            metrics = view.rename(columns={"Position Ticker": "position_ticker"})
            metrics = _attach_return_columns(metrics, ticker_col="position_ticker")
            metrics = metrics.rename(columns={"position_ticker": "Position Ticker"})

            ordered_cols = [
                "Fund Name","Position Name","Position Ticker","Position Sector",
                "Position Weight (%)","Report Date","MTD %","YTD %"
            ]
            metrics = metrics[ordered_cols].sort_values(["MTD %"], ascending=[False])

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Report Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD", step="day"),
                    "Position Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "MTD %": st.column_config.NumberColumn(format="%.2f%%"),
                    "YTD %": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

    with top_tabs[1]:
        df = letters.copy()
        for col in ["position_thesis_summary","position_ticker"]:
            if col not in df.columns:
                df[col] = None
        mask = _nonnull_mask(df["position_thesis_summary"]) & _nonnull_mask(df["position_ticker"])
        df = df[mask].copy()

        for col in ["position_name","position_sector","position_duration_view"]:
            if col not in df.columns:
                df[col] = None

        # Build display view
        view = df[[
            "fund_name","report_date","position_name","position_ticker","position_sector",
            "position_thesis_summary","position_duration_view"
        ]].rename(columns={
            "fund_name":"Fund Name",
            "report_date":"Report Date",
            "position_name":"Position Name",
            "position_ticker":"Position Ticker",
            "position_sector":"Position Sector",
            "position_thesis_summary":"Position Thesis Summary",
            "position_duration_view":"Position Duration View",
        })

        # Normalize display date and create calc date
        view["Report Date"] = pd.to_datetime(view["Report Date"], errors="coerce")
        metrics = view.rename(columns={"Position Ticker":"position_ticker"}).copy()
        metrics["report_date_dt"] = pd.to_datetime(view["Report Date"], errors="coerce", utc=True).dt.tz_localize(None)

        # Attach only MTD/YTD (per earlier change)
        metrics = _attach_return_columns(metrics, ticker_col="position_ticker")

        # Attach Since Report %
        metrics = _attach_since_report_col(
            metrics,
            ticker_col="position_ticker",
            report_date_col="report_date_dt",
            colname="Since Report %",
        )

        # Restore display names, drop calc helper
        metrics = metrics.rename(columns={"position_ticker":"Position Ticker"})
        metrics = metrics.drop(columns=["report_date_dt"], errors="ignore")

        # Order and render
        metrics = metrics.sort_values("Report Date", ascending=False)
        ordered_cols = [
            "Fund Name","Report Date","Position Name","Position Ticker","Position Sector",
            "Position Thesis Summary","Since Report %","MTD %","YTD %","Position Duration View",
        ]
        metrics = metrics[ordered_cols]

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Report Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD", step="day"),
                "Since Report %": st.column_config.NumberColumn(format="%.2f%%"),
                "MTD %": st.column_config.NumberColumn(format="%.2f%%"),
                "YTD %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    
    # ── Row 2 ─────────────────────────────────────────────────────────────
    st.write("## Market Views")
    st.write("Latest manager insights extracted from fund letters, factsheets and external research communications")

    bottom_tabs = st.tabs([
        "Fund Insights",
        "External Research"
    ])

    with bottom_tabs[0]:
        df = letters.copy()
        if "macro_view_category" not in df.columns:
            st.warning("Column 'macro_view_category' not found in letters.")
            df = pd.DataFrame(columns=["Macro Category","Fund Name","Report Date","Macro View Summary","Macro View Direction"])
        else:
            df = df[_nonnull_mask(df["macro_view_category"])].copy()
            for c in ["macro_view_summary","macro_view_direction"]:
                if c not in df.columns:
                    df[c] = None
            df = df[["macro_view_category","fund_name","report_date","macro_view_summary","macro_view_direction"]]
            df = _parse_report_date(df, "report_date")
            df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            df = df.rename(columns={
                "macro_view_category": "Macro Category",
                "fund_name": "Fund Name",
                "report_date": "Report Date",
                "macro_view_summary": "Macro View Summary",
                "macro_view_direction": "Macro View Direction",
            }).sort_values("Report Date", ascending=False)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Report Date": st.column_config.DatetimeColumn(format="YYYY-MM-DD", step="day"),
            },
        )


    # 4) Legacy “External Research” preserved from original Market Views
    with bottom_tabs[1]:
        # existing implementation preserved
        if not ("market_views" in st.secrets and "sheet_id" in st.secrets["market_views"]):
            st.error("Missing 'market_views' configuration in secrets.")
            return
        sheet_id = st.secrets["market_views"].get("sheet_id")
        worksheet = st.secrets["market_views"].get("worksheet", "Sheet1")
        df_mv = load_sheet(sheet_id, worksheet)
        if df_mv.empty:
            st.warning("No data returned from the market views sheet.")
            return
        if "Date" in df_mv.columns:
            df_mv["Date"] = pd.to_datetime(df_mv["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        cols_to_hide = [
            "Document Type","Data & Evidence","Key Themes","Risks/Uncertainties",
            "Risks / Uncertainties","Evidence Strength & Uniqueness","Evidence Strenght & Uniqueness",
            "Follow-up Actions","Title"
        ]
        df_display = df_mv.drop(columns=[c for c in cols_to_hide if c in df_mv.columns], errors="ignore")
        desired_order = ["Asset Class & Region","Date","Author(s)","Institution/Source","Market Regime/Context","Instrument Name"]
        columns = [c for c in desired_order if c in df_display.columns]
        columns += [c for c in df_display.columns if c not in columns]
        df_display = df_display[columns]

        filter_columns = st.multiselect("Columns to filter", df_display.columns.tolist(), key="mv_filters")
        filtered = df_display.copy()
        for col in filter_columns:
            choices = sorted(df_display[col].dropna().unique().tolist())
            selected = st.multiselect(f"{col}", choices, key=f"mv_{col}")
            if selected:
                filtered = filtered[filtered[col].isin(selected)]
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

        def get_full_row(row):
            mask = pd.Series([True] * len(df_mv))
            for key in ["Date","Asset Class & Region","Institution/Source"]:
                if key in df_mv.columns and key in row:
                    mask &= (df_mv[key] == row.get(key))
            matches = df_mv[mask]
            if not matches.empty:
                return matches.iloc[0]
            return row

        if isinstance(selected, list) and len(selected) > 0:
            row = selected[0]
            full_row = get_full_row(row)
            with st.expander("Market View Details", expanded=True):
                all_columns = list(filtered.columns) + [c for c in cols_to_hide if c in df_mv.columns]
                cols = st.columns(2)
                for i, col in enumerate(all_columns):
                    with cols[i % 2]:
                        st.markdown(f"**{col}:**")
                        value = full_row[col] if col in full_row else row.get(col)
                        st.markdown(value if pd.notna(value) else "_(empty)_")


# --- Shared utilities (existing + minor helpers) ---
def parse_any_date(val):
    if pd.isna(val) or not str(val).strip():
        return None
    s = str(val).strip()
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if pd.notna(dt):
        if dt.day == 1 and (
            s.lower().endswith(str(dt.year)) and (
                s.lower().startswith(dt.strftime("%B").lower()) or
                s.lower().startswith(dt.strftime("%b").lower()) or
                "-" in s or "/" in s
            )
        ):
            if (
                len(s.split()) == 2 and
                not any(char.isdigit() for char in s.split()[0])
            ) or s.count("-") == 1 or s.count("/") == 1:
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                dt = dt.replace(day=last_day)
        return dt.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(s + " 1", errors="coerce")
        if pd.notna(dt):
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            dt = dt.replace(day=last_day)
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

def percent_to_float(val):
    try:
        if pd.isna(val):
            return None
        return float(str(val).replace("%", "").strip())
    except Exception:
        return None

def aum_to_float(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower().replace(",", "")
    # fast path
    x = pd.to_numeric(s.replace("$", ""), errors="coerce")
    if pd.notna(x):
        return float(x)
    # unit parsing
    import re
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None
    num = float(m.group(1))
    if "bn" in s or s.endswith(" b") or s.endswith("b"):
        return num * 1e9
    if "mm" in s or "mn" in s or "m" in s:
        return num * 1e6
    if "k" in s:
        return num * 1e3
    return num


# ADD near other helpers (below percent_to_float)
import re

def _split_bullets(text: str) -> list[str]:
    s = ("" if text is None else str(text)).strip()
    if not s:
        return []

    # Normalize common separators into line breaks
    s = s.replace("•", "\n• ")
    s = re.sub(r"\s+(\d{1,2}[\.\)])\s+", r"\n\1 ", s)  # "1. ..." or "2) ..."

    # Split into lines and strip leading list tokens
    lines = [ln for ln in (x.strip() for x in s.splitlines()) if ln]
    items = []
    for ln in lines:
        ln = re.sub(r"^\s*(?:•|-|\(?\d{1,3}[\.\)])\s*", "", ln).strip()
        if ln:
            items.append(ln)

    # Fallback: still one blob with inline numbering
    if len(items) == 1 and re.search(r"\b\d{1,2}[\.\)]\s+", items[0]):
        parts = re.split(r"\s(?=\d{1,2}[\.\)]\s+)", items[0])
        items = [re.sub(r"^\s*(?:\(?\d{1,2}[\.\)])\s*", "", p).strip() for p in parts]
        items = [p for p in items if p]

    return items

# replace your helper signature
def _format_exposure_table(df: pd.DataFrame) -> Styler:
    out = df.copy()
    out.index = out.index.astype(str).str.strip().str.strip('"').str.strip("'")
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    net_cols = [c for c in out.columns if c.endswith("_net")]
    styler = out.style.format("{:.2f}%")
    if net_cols:
        styler = styler.set_properties(subset=pd.IndexSlice[:, [net_cols[-1]]], **{"font-weight": "bold"})
    return styler


# ======== show_fund_monitor()  ========
# === helpers for robust profile lookup + safe rendering ===
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _norm_text(s) -> str:
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "", str(s).lower()).strip()

def _lookup_profile(db: pd.DataFrame, fund_choice: str, selected_canonical_id: str):
    from difflib import SequenceMatcher
    db = _clean_cols(db)

    # 1) match by canonical id if sheet carries it
    for cid_col in ["canonical_id", "fund_id", "Canonical ID", "Fund ID"]:
        if cid_col in db.columns:
            m = db[cid_col].astype(str) == str(selected_canonical_id)
            if m.any():
                return db[m].iloc[0]

    # 2) exact/ci match on Fund Name
    if "Fund Name" in db.columns:
        m = db["Fund Name"].astype(str).str.strip() == str(fund_choice).strip()
        if not m.any():
            m = db["Fund Name"].astype(str).str.casefold() == str(fund_choice).casefold()
        if m.any():
            return db[m].iloc[0]

        # 3) fuzzy fallback
        key = _norm_text(fund_choice)
        keys = db["Fund Name"].map(_norm_text)
        if not keys.empty:
            sim = keys.apply(lambda k: SequenceMatcher(a=k, b=key).ratio())
            idx = sim.idxmax()
            if float(sim.loc[idx]) >= 0.90:
                return db.loc[idx]

    return None

# logical->column aliases so sheets with slightly different headers still populate
_ALIASES = {
    "Fund Name": ["Fund Name", "Name", "Fund"],
    "Asset Class": ["Asset Class"],
    "Type": ["Type", "Strategy Type"],
    "Summary": ["Summary", "One-Liner", "Description"],
    "AUM (in USD Millions)": ["AUM (in USD Millions)", "AUM (USD m)", "AUM (USD Millions)", "AUM"],
    "Time Horizon": ["Time Horizon", "Horizon"],
    "Style": ["Style"],
    "Geo": ["Geo", "Geography", "Region"],
    "Sector": ["Sector", "Primary Sector", "Focus Sector"],
    "Avg # Positions": ["Avg # Positions", "Average # Positions", "Avg Positions"],
    "Avg Gross": ["Avg Gross", "Average Gross"],
    "Avg Net": ["Avg Net", "Average Net"],
    "Management Fee": ["Management Fee", "Mgmt Fee"],
    "Performance Fee": ["Performance Fee", "Perf Fee", "Incentive Fee"],
    "Inception": ["Inception", "Inception Year", "Launch"],
    "Market Opportunity": ["Market Opportunity", "Opportunity"],
    "Risks": ["Risks", "Key Risks"],
    "Team Background": ["Team Background", "Team", "Background"],
    "Edge / What they do": ["Edge / What they do", "Edge", "What they do"],
    "Portfolio": ["Portfolio", "Guidelines"],
}

def _get_val(row: pd.Series | None, logical_col: str, fallback: str = "") -> str:
    if row is None:
        return fallback
    for col in _ALIASES.get(logical_col, [logical_col]):
        if col in row.index:
            v = row.get(col)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s != "":
                return s
    return fallback

def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed/object cols to Arrow-safe types before st.dataframe."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_object_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() >= 0.9 * len(s):
                out[c] = num
            else:
                out[c] = s.where(~s.isna(), "").astype("string")
        elif pd.api.types.is_integer_dtype(s):
            out[c] = s.astype("Int64")
    return out

# ======== REPLACE the whole function: show_fund_monitor() ========
def show_fund_monitor() -> None:
    import pandas as pd
    import numpy as np
    import streamlit as st
    import altair as alt

    # ========= Data loads =========
    if not ("exposures" in st.secrets and "sheet_id" in st.secrets["exposures"]):
        st.error("Missing 'exposures' configuration in secrets.")
        return
    exp_sheet_id = st.secrets["exposures"]["sheet_id"]
    exp_ws = st.secrets["exposures"].get("worksheet", "Sheet1")
    df = load_sheet(exp_sheet_id, exp_ws)
    if df.empty:
        st.warning("No data returned from the exposures sheet.")
        return

    if not ("securities_master" in st.secrets and "sheet_id" in st.secrets["securities_master"]):
        st.error("Missing 'securities_master' configuration in secrets.")
        return
    sec_sheet_id = st.secrets["securities_master"]["sheet_id"]
    sec_ws = st.secrets["securities_master"].get("worksheet", "Sheet1")
    sec_df = load_sheet(sec_sheet_id, sec_ws)
    if sec_df.empty or not {"canonical_id", "canonical_name"} <= set(sec_df.columns):
        st.error("Missing columns in securities_master.")
        return

    # ========= Fund select =========
    exposure_fund_ids = set(df["fund_id"].dropna().astype(str).unique()) if "fund_id" in df.columns else set()
    canonical_funds = (
        sec_df[sec_df["canonical_id"].astype(str).isin(exposure_fund_ids)][["canonical_id", "canonical_name"]]
        .drop_duplicates()
        .sort_values("canonical_name")
    )
    if canonical_funds.empty:
        st.warning("No matching funds between exposures and securities_master.")
        return

    default_fund = st.secrets.get("defaults", {}).get("fund", None)
    fund_options = canonical_funds["canonical_name"].tolist()
    fund_index = fund_options.index(default_fund) if default_fund in fund_options else 0

    csel1, _ = st.columns([2, 1])
    with csel1:
        fund_choice = st.selectbox("Select Fund", fund_options, index=fund_index, key="fm_fund_select")
    selected_canonical_id = canonical_funds.loc[
        canonical_funds["canonical_name"] == fund_choice, "canonical_id"
    ].iloc[0]

    # Normalize dates now; pre-slice to this fund
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    fund_df = df[df["fund_id"].astype(str) == str(selected_canonical_id)].copy()
    if fund_df.empty:
        st.warning("No records found for the selected fund.")
        return

    # ========= Tabs =========
    tabs = st.tabs(["Overview", "Exposures", "Quant"])

    # --------------------------------
    # Tab 1: Overview
    # --------------------------------
    # ====================== OVERVIEW TAB (drop-in replacement) ======================
    with tabs[0]:
        # ---- Load profile row from Fund Database ----
        profile_row = None
        try:
            if "fund_database" in st.secrets and "sheet_id" in st.secrets["fund_database"]:
                db_id = st.secrets["fund_database"]["sheet_id"]
                db_ws = st.secrets["fund_database"].get("worksheet", "fund database")
                db = load_sheet(db_id, db_ws)
                if not db.empty:
                    profile_row = _lookup_profile(db, fund_choice, str(selected_canonical_id))
        except Exception:
            profile_row = None

        # ---- Helpers: scorecards + text rendering ----
        FIELD_DESC = {
            "Asset Class": "Primary investment universe.",
            "Type": "Directional posture or mandate.",
            "AUM (in USD Millions)": "Strategy size in USD millions.",
            "Time Horizon": "Typical holding period.",
            "Style": "Investment style tilt.",
            "Geo": "Main geography focus.",
            "Sector": "Primary sector focus.",
            "Avg # Positions": "Average number of active positions.",
            "Avg Gross": "Typical gross exposure.",
            "Avg Net": "Typical net exposure.",
            "Management Fee": "Annual management fee.",
            "Performance Fee": "Incentive fee on profits.",
            "Inception": "Strategy start year/date.",
        }

        def _scorecard(label: str, value: str, desc: str = "") -> None:
            v = value if (value is not None and str(value).strip() != "") else "-"
            d = desc or "&nbsp;"
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;height:100%;background:#ffffff;">
                <div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em">{label}</div>
                <div style="font-size:20px;font-weight:600;margin-top:4px">{v}</div>
                <div style="font-size:12px;color:#9ca3af;margin-top:6px">{d}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ===== Top row: Fund Name + Summary =====
        t1, t2 = st.columns([1, 2])
        with t1:
            st.markdown(
                f"<div style='font-size:28px;font-weight:700;line-height:1.1'>{_get_val(profile_row, 'Fund Name', fund_choice)}</div>",
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(_md_text(_get_val(profile_row, "Summary")), unsafe_allow_html=True)

        # ===== Scorecards with descriptions =====
        c1, c2 = st.columns(2)
        with c1: _scorecard("Asset Class", _get_val(profile_row, "Asset Class"), FIELD_DESC["Asset Class"])
        with c2: _scorecard("Type", _get_val(profile_row, "Type"), FIELD_DESC["Type"])

        c3, c4, c5, c6, c7 = st.columns(5)
        with c3: _scorecard("Size", _get_val(profile_row, "AUM (in USD Millions)"), FIELD_DESC["AUM (in USD Millions)"])
        with c4: _scorecard("Time Horizon", _get_val(profile_row, "Time Horizon"), FIELD_DESC["Time Horizon"])
        with c5: _scorecard("Style", _get_val(profile_row, "Style"), FIELD_DESC["Style"])
        with c6: _scorecard("Geo", _get_val(profile_row, "Geo"), FIELD_DESC["Geo"])
        with c7: _scorecard("Sector", _get_val(profile_row, "Sector"), FIELD_DESC["Sector"])

        c8, c9, c10, c11, c12, c13 = st.columns(6)
        with c8: _scorecard("Avg # Positions", _get_val(profile_row, "Avg # Positions"), FIELD_DESC["Avg # Positions"])
        with c9: _scorecard("Avg Gross", _get_val(profile_row, "Avg Gross"), FIELD_DESC["Avg Gross"])
        with c10: _scorecard("Avg Net", _get_val(profile_row, "Avg Net"), FIELD_DESC["Avg Net"])
        with c11: _scorecard("Management Fee", _get_val(profile_row, "Management Fee"), FIELD_DESC["Management Fee"])
        with c12: _scorecard("Performance Fee", _get_val(profile_row, "Performance Fee"), FIELD_DESC["Performance Fee"])
        with c13: _scorecard("Inception", _get_val(profile_row, "Inception"), FIELD_DESC["Inception"])

        # ===== Narrative sections (expanders, expanded by default; exact sheet labels) =====
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            with st.expander("Market Opportunity", expanded=True):
                st.markdown(_md_text(_get_val(profile_row, "Market Opportunity")), unsafe_allow_html=True)
        with r1c2:
            with st.expander("Risks", expanded=True):
                st.markdown(_md_text(_get_val(profile_row, "Risks")), unsafe_allow_html=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            with st.expander("Team Background", expanded=True):
                st.markdown(_md_text(_get_val(profile_row, "Team Background")), unsafe_allow_html=True)
        with r2c2:
            with st.expander("Edge / What they do", expanded=True):
                st.markdown(_md_text(_get_val(profile_row, "Edge / What they do")), unsafe_allow_html=True)
        with r2c3:
            with st.expander("Portfolio", expanded=True):
                st.markdown(_md_text(_get_val(profile_row, "Portfolio")), unsafe_allow_html=True)

        st.markdown("---")

        # ===== Cumulative Return + AUM history =====
        hc1, hc2 = st.columns(2)
        with hc1:
            st.subheader("Cumulative Performance")
            track_record = fetch_track_record_json(selected_canonical_id)
            if track_record and isinstance(track_record.get("returns"), list):
                returns_df = pd.DataFrame(track_record["returns"])
                if "date" not in returns_df.columns or "return" not in returns_df.columns:
                    if {"date", "ret"} <= set(returns_df.columns):
                        returns_df = returns_df.rename(columns={"ret": "return"})
                    elif {"date", "value"} <= set(returns_df.columns):
                        returns_df = returns_df.rename(columns={"value": "return"})
                if {"date", "return"} <= set(returns_df.columns):
                    returns_df["date"] = pd.to_datetime(returns_df["date"], errors="coerce")
                    returns_df["return"] = pd.to_numeric(returns_df["return"], errors="coerce")
                    returns_df = returns_df.dropna(subset=["date", "return"]).sort_values("date")
                    if not returns_df.empty:
                        returns_df["cum"] = (1.0 + returns_df["return"]).cumprod() - 1.0
                        ch = (
                            alt.Chart(returns_df)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("date:T", title="Date"),
                                y=alt.Y("cum:Q", title="Cumulative Return", axis=alt.Axis(format="~%")),
                                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("cum:Q", title="Cumulative", format=".2%")],
                            )
                            .properties(height=350)
                        )
                        st.altair_chart(ch, use_container_width=True)
                else:
                    st.info("No return series available.")
            else:
                st.info("No return series available.")

        with hc2:
            st.subheader("Historical AUM")
            aum_hist = fund_df.copy()
            aum_hist["date"] = pd.to_datetime(aum_hist["date"], errors="coerce")
            aum_hist["aum_fund_num"] = aum_hist.get("aum_fund", np.nan).apply(aum_to_float) if "aum_fund" in aum_hist.columns else np.nan
            aum_hist = (
                aum_hist.dropna(subset=["date", "aum_fund_num"])
                .sort_values("date")
                .drop_duplicates(subset=["date"], keep="last")
                .rename(columns={"aum_fund_num": "AUM"})
            )
            if aum_hist.empty:
                st.info("No AUM history available.")
            else:
                ch = (
                    alt.Chart(aum_hist)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("AUM:Q", title="AUM", axis=alt.Axis(format=",.0f")),
                        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("AUM:Q", title="AUM", format=",.0f")],
                    )
                    .properties(height=350)
                )
                st.altair_chart(ch, use_container_width=True)
    # ==================== END OVERVIEW TAB ====================




# =========================
# Market Analytics Section
# =========================

# 1) Monthly Seasonality Explorer
def _fetch_history(ticker: str, start: str = "1928-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # If yfinance returned MultiIndex columns (e.g., single ticker under a second level),
    # collapse to a single set of OHLCV columns.
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        if isinstance(ticker, str) and ticker in set(lvl1):
            df = df.xs(ticker, axis=1, level=1)
        else:
            # fallback to the first symbol present
            first = next(iter(dict.fromkeys(lvl1)))  # preserves order
            df = df.xs(first, axis=1, level=1)

    # Ensure there is a concrete 'date' column
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "date"
    df = df.reset_index()

    # Normalize headers
    df.columns = [str(c).lower() for c in df.columns]

    # Guarantee 'adj close'
    if "adj close" not in df.columns:
        if "close" in df.columns:
            df["adj close"] = df["close"]
        elif "adjclose" in df.columns:
            df["adj close"] = df["adjclose"]

    # Final type assurance
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _seasonality_stats(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()
    if "adj close" not in px.columns and "close" in px.columns:
        px["adj close"] = px["close"]
    if "date" not in px.columns:
        raise ValueError("History frame missing 'date' after fetch.")
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"])

    px["ret"] = px["adj close"].pct_change()
    px["year"] = px["date"].dt.year
    px["month"] = px["date"].dt.month
    m = px.groupby(["year","month"]).agg(rv=("ret","sum")).dropna().reset_index()
    stats = m.groupby("month").agg(
        median_month_return=("rv","median"),
        hit_rate=("rv", lambda x: (x > 0).mean())
    ).reset_index()
    stats["median_month_return_pct"] = stats["median_month_return"] * 100
    stats["hit_rate_pct"] = stats["hit_rate"] * 100
    return stats

def _monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"]).sort_values("date")
    # month-end close then MoM return
    px["m"] = px["date"].dt.to_period("M")
    me = px.groupby("m", as_index=False)["adj close"].last().rename(columns={"adj close":"close_me"})
    me["ret"] = me["close_me"].pct_change()
    me["year"] = me["m"].dt.year
    me["month"] = me["m"].dt.month
    return me.dropna(subset=["ret"])

def _seasonality_summary(me: pd.DataFrame) -> pd.DataFrame:
    g = me.groupby("month")["ret"]
    out = pd.DataFrame({
        "median": g.median(),
        "q1": g.quantile(0.25),
        "q3": g.quantile(0.75),
        "low": g.min(),
        "high": g.max(),
        "hit": (g.apply(lambda s: (s > 0).mean()) * 100.0)
    }).reset_index()
    out["month_name"] = pd.to_datetime(out["month"], format="%m").dt.strftime("%b")
    return out.sort_values("month")

# Replace the whole function
def view_monthly_seasonality(ticker_override: str | None = None, start_year_override: int | None = None):
    st.subheader("Monthly Seasonality Explorer")

    # shared controls or overrides
    if ticker_override is None or start_year_override is None:
        c1, c2 = st.columns([2,1])
        with c1:
            ticker = st.text_input("Ticker (Yahoo Finance)", value="SPY", key="seas_ticker")
        with c2:
            start_year = st.number_input("Start Year", min_value=1900, max_value=datetime.now().year, value=1990, key="seas_start")
    else:
        ticker = ticker_override
        start_year = start_year_override

    if not ticker:
        return

    df = _fetch_history(ticker, start=f"{start_year}-01-01")
    me = _monthly_returns(df)
    stats = _seasonality_summary(me)

    # compute display vectors
    months = stats["month_name"].tolist()
    x = np.arange(len(months))
    med = (stats["median"] * 100.0).to_numpy()
    low = (stats["low"] * 100.0).to_numpy()
    high = (stats["high"] * 100.0).to_numpy()
    q1 = (stats["q1"] * 100.0).to_numpy()
    q3 = (stats["q3"] * 100.0).to_numpy()
    hit = stats["hit"].to_numpy()

    # colors by sign of median
    colors = np.where(med >= 0, "#2ca02c", "#d62728")  # green/red

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    # bar for median
    ax.bar(x, med, width=0.6, color=colors, alpha=0.85, zorder=2)

    # interquartile box overlay (q1–q3) to mimic repo style
    ax.vlines(x, q1, q3, colors="gray", linewidth=8, alpha=0.25, zorder=1)

    # whiskers min–max
    yerr = np.vstack([med - low, high - med])
    ax.errorbar(x, med, yerr=yerr, fmt="none", ecolor="lightgray", elinewidth=2, capsize=3, zorder=3)

    # secondary axis for hit rate with black diamond markers
    ax2 = ax.twinx()
    ax2.plot(x, hit, marker="D", linestyle="None", color="black", markersize=6, zorder=4)

    # axes formatting
    ax.set_xticks(x, months)
    ax.set_ylabel("Median return (%)")
    ax2.set_ylabel("Hit rate of positive returns")
    ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(lambda v, pos: f"{int(v):d}%")
    ax.axhline(0, color="#888", linewidth=1, zorder=0)

    # title span
    yr_end = int(me["year"].max()) if not me.empty else datetime.now().year
    ax.set_title(f"{ticker} Seasonality ({start_year}-{yr_end})", pad=12, fontweight="bold")

    # best/worst banner (by median)
    best_i = int(np.nanargmax(med))
    worst_i = int(np.nanargmin(med))
    best_txt = f"Best month: {months[best_i]} ({med[best_i]:.2f}%) | High {high[best_i]:.2f}% | Low {low[best_i]:.2f}%"
    worst_txt = f"Worst month: {months[worst_i]} ({med[worst_i]:.2f}%) | High {high[worst_i]:.2f}% | Low {low[worst_i]:.2f}%"
    st.markdown(f"**{best_txt}**  **{worst_txt}**")

    st.pyplot(fig, clear_figure=True)


# 2) Market Memory Explorer
def _ytd_path(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()
    px["ret"] = px["adj close"].pct_change().fillna(0.0)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["year"] = px["date"].dt.year
    px["doy"] = px["date"].dt.dayofyear
    px["cum"] = (1.0 + px["ret"]).groupby(px["year"]).cumprod() - 1.0
    return px[["doy", "year", "cum"]]

def _closest_years(cur: pd.Series, hist: pd.DataFrame, k: int=5) -> List[int]:
    # Align by day-of-year, compute Euclidean distance
    merged = hist.pivot(index="doy", columns="year", values="cum").dropna(how="all")
    common = merged.index[merged.index.isin(cur.index)]
    if len(common) < 5:
        return []
    diffs = {}
    for yr in merged.columns:
        v = merged.loc[common, yr].fillna(method="ffill").fillna(0.0)
        diffs[yr] = float(np.sqrt(((v - cur.loc[common])**2).mean()))
    ranked = sorted(diffs.items(), key=lambda x: x[1])
    res = [y for y,_ in ranked if y != datetime.now().year][:k]
    return res

# Replace the whole function
def view_market_memory(ticker_override: str | None = None, start_year_override: int | None = None, k_override: int | None = None):
    st.subheader("Market Memory Explorer")
    if ticker_override is None or start_year_override is None or k_override is None:
        l, r = st.columns([2,1])
        with l:
            ticker = st.text_input("Ticker (Yahoo Finance)", value="SPY", key="mm_ticker")
        with r:
            start_year = st.number_input("History start year", min_value=1900, max_value=datetime.now().year, value=1950, key="mm_start")
            k = st.slider("Similar years", min_value=3, max_value=10, value=5, key="mm_k")
    else:
        ticker = ticker_override
        start_year = start_year_override
        k = k_override

    if not ticker:
        return
    df = _fetch_history(ticker, start=f"{start_year}-01-01")
    px = df.copy()
    if "adj close" not in px.columns and "close" in px.columns:
        px["adj close"] = px["close"]
    px["ret"] = px["adj close"].pct_change().fillna(0.0)
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["doy"] = px["date"].dt.dayofyear
    px["year"] = px["date"].dt.year
    px["cum"] = (1.0 + px["ret"]).groupby(px["year"]).cumprod() - 1.0

    this_year = px[px["year"] == datetime.now().year].set_index("doy")["cum"]
    hist = _ytd_path(df)

    pivot = hist.pivot(index="doy", columns="year", values="cum")
    common = pivot.index[pivot.index.isin(this_year.index)]
    diffs = {
        yr: float(np.sqrt(((pivot.loc[common, yr].ffill().fillna(0.0) - this_year.loc[common])**2).mean()))
        for yr in pivot.columns if yr != datetime.now().year
    }
    years = [y for y, _ in sorted(diffs.items(), key=lambda x: x[1])[:k]]

    import altair as alt
    current_year = datetime.now().year

    px["year_str"] = px["year"].astype(str)
    subset = px[px["year"].isin(years + [current_year])].copy()

    chart = (
        alt.Chart(subset)
        .mark_line()
        .encode(
            x=alt.X("doy:Q", title="Day of Year"),
            y=alt.Y("cum:Q", title="Cumulative Return"),
            color=alt.Color(
                "year_str:N",
                title="Year",
                legend=alt.Legend(orient="top", direction="horizontal", columns=8),
            ),
            size=alt.condition(alt.datum.year == current_year, alt.value(3.0), alt.value(1.5)),
            opacity=alt.condition(alt.datum.year == current_year, alt.value(1.0), alt.value(0.45)),
            tooltip=[
                alt.Tooltip("year_str:N", title="Year"),
                alt.Tooltip("doy:Q", title="Day"),
                alt.Tooltip("cum:Q", title="Cumulative", format=".2%"),
            ],
        )
        .properties(height=520, padding={"left": 8, "right": 80, "top": 10, "bottom": 30})
        .configure_legend(labelLimit=2000)
    )

    st.altair_chart(
        chart,
        use_container_width=True,
        key=f"mm_chart_{ticker}_{start_year}_{k}_{current_year}",
    )


# 3) Breakout Scanner
def _rolling_high(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()

def _scan_breakouts(tickers: List[str], lookbacks: List[int]) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=max(lookbacks) * 3)
    # Auto-adjusted close via yfinance; returns MultiIndex columns for multi-ticker
    px = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )
    # Extract the Close level and ensure a 2D frame
    if isinstance(px.columns, pd.MultiIndex) and "Close" in px.columns.levels[1]:
        data = px.xs("Close", axis=1, level=1)
    else:
        # Single ticker case
        data = px[["Close"]] if "Close" in px.columns else px
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    latest = data.iloc[-1]
    out = []
    for t in data.columns:
        series = data[t].astype(float)
        row = {"ticker": t, "price": float(latest[t])}
        for n in lookbacks:
            rh = float(series.rolling(n, min_periods=n).max().iloc[-1])
            row[f"high_{n}d"] = rh
            row[f"breakout_{n}d"] = bool(latest[t] >= rh)
        out.append(row)
    return pd.DataFrame(out).sort_values("ticker")


def view_breakout_scanner():
    st.subheader("Breakout Scanner")
    default = ["SPY","QQQ","IWM","AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"]
    universe = st.text_area("Universe (comma-separated tickers)", value=",".join(default))
    lookbacks = st.multiselect("High lookbacks", [20,50,100,200], default=[20,50,100,200])
    if not universe.strip():
        return
    tickers = [t.strip().upper() for t in universe.split(",") if t.strip()]
    df = _scan_breakouts(tickers, lookbacks)
    bool_cols = [c for c in df.columns if c.startswith("breakout_")]
    for c in bool_cols:
        df[c] = df[c].map({True:"Yes", False:"No"})
    st.dataframe(df, use_container_width=True)

# 4) 10-Year Nominal & Real Yield Dashboard
# FRED series:
# - Nominal: DGS10
# - Real: REAINTRATREARAT10Y (10Y real rate, alt: DFII10 TIPS yield)
def _fred(series: str, start: str="1990-01-01") -> pd.DataFrame:
    s = pdr.DataReader(series, "fred", start=start)
    s = s.rename(columns={series:"value"}).reset_index().rename(columns={"DATE":"date"})
    return s

def view_real_yield_dashboard():
    st.subheader("10-Year Nominal and Real Yield")
    s_nom = _fred("DGS10", start="1990-01-01")
    try:
        s_real = _fred("REAINTRATREARAT10Y", start="1990-01-01")
    except Exception:
        # fallback to TIPS 10Y
        s_real = _fred("DFII10", start="2003-01-01")
    joined = pd.merge_asof(
        s_real.sort_values("date"),
        s_nom.sort_values("date"),
        on="date", direction="backward", suffixes=("_real","_nom")
    ).dropna()
    joined["real_mom_63d"] = joined["value_real"].diff(63)
    import altair as alt
    c1 = alt.Chart(joined).mark_line().encode(x="date:T", y=alt.Y("value_nom:Q", title="Nominal 10Y (%)"))
    c2 = alt.Chart(joined).mark_line(color="#d62728").encode(x="date:T", y=alt.Y("value_real:Q", title="Real 10Y (%)"))
    c3 = alt.Chart(joined).mark_line().encode(x="date:T", y=alt.Y("real_mom_63d:Q", title="63D Δ Real 10Y"))
    st.altair_chart(c1, use_container_width=True)
    st.altair_chart(c2, use_container_width=True)
    st.altair_chart(c3, use_container_width=True)
    st.caption("Sources: FRED DGS10 (nominal), REAINTRATREARAT10Y (real). Real-yield momentum = 63-trading-day change. See repo notes for interpretation.")

# 5) Liquidity & Fed Policy Tracker
# Net Liquidity = WALCL − RRP − TGA (units: $bn). RRP series: RRPONTSYD; TGA: WTREGEN.
def view_liquidity_tracker():
    st.subheader("Liquidity & Fed Policy Tracker")
    walcl = _fred("WALCL", start="2015-01-01")      # Fed balance sheet (weekly)
    rrp = _fred("RRPONTSYD", start="2015-01-01")    # Overnight RRP
    tga = _fred("WTREGEN", start="2015-01-01")      # Treasury General Account
    # Align to business days (forward-fill)
    for s in (walcl, rrp, tga):
        s["date"] = pd.to_datetime(s["date"])
    dt_index = pd.DataFrame({"date": pd.date_range(min(walcl["date"].min(), rrp["date"].min(), tga["date"].min()),
                                                   datetime.now(), freq="B")})
    def align(df): 
        return pd.merge_asof(dt_index, df.sort_values("date"), on="date", direction="backward")
    a_w, a_r, a_t = align(walcl), align(rrp), align(tga)
    liq = dt_index.copy()
    liq["walcl"] = a_w["value"]
    liq["rrp"] = a_r["value"]
    liq["tga"] = a_t["value"]
    liq["net_liquidity"] = liq["walcl"] - liq["rrp"] - liq["tga"]
    # Optional SPX overlay
    try:
        spy = _fetch_history("SPY", start="2015-01-01")[["date","adj close"]].rename(columns={"adj close":"spy"})
        spy["date"] = pd.to_datetime(spy["date"], errors="coerce")
        liq = pd.merge_asof(liq.sort_values("date"), spy.sort_values("date"), on="date", direction="backward")
    except Exception:
        pass
    import altair as alt
    left = alt.Chart(liq).mark_line().encode(x="date:T", y=alt.Y("net_liquidity:Q", title="Net Liquidity ($bn)"))
    st.altair_chart(left, use_container_width=True)
    if "spy" in liq.columns:
        right = alt.Chart(liq).mark_line(color="#2ca02c").encode(x="date:T", y=alt.Y("spy:Q", title="SPY (Adj Close)"))
        st.altair_chart(right, use_container_width=True)
    st.caption("Definition: Net Liquidity = WALCL − RRP − TGA; sustained rises often align with risk-on conditions.")

# 6) Market Stress Composite
# Components: VIX (^VIX), HY OAS (BAMLH0A0HYM2), Curve (T10Y2Y), SPY drawdown.
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

def view_market_stress():
    st.subheader("Market Stress Composite")
    # Fetch
    vix = _fetch_history("^VIX", start="2004-01-01")[["date","adj close"]].rename(columns={"adj close":"vix"})
    hy = _fred("BAMLH0A0HYM2", start="2004-01-01").rename(columns={"value":"hy_oas"})
    curve = _fred("T10Y2Y", start="2004-01-01").rename(columns={"value":"t10y2y"})
    spy = _fetch_history("SPY", start="2004-01-01")[["date","adj close"]].rename(columns={"adj close":"spy"})

    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    hy["date"] = pd.to_datetime(hy["date"], errors="coerce")
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    spy["date"] = pd.to_datetime(spy["date"], errors="coerce")

    # Merge
    df = vix.merge(hy, on="date", how="outer").merge(curve, on="date", how="outer").merge(spy, on="date", how="outer").sort_values("date")
    df = df.ffill().dropna()
    # Drawdown
    roll_max = df["spy"].cummax()
    dd = (df["spy"] / roll_max) - 1.0
    # Normalize components to [0,1] stress direction (higher = more stress)
    z_vix = _pct_rank(df["vix"])
    z_hy = _pct_rank(df["hy_oas"])
    z_curve = 1 - _pct_rank(df["t10y2y"])  # more inverted curve -> higher stress
    z_dd = _pct_rank(-dd)  # deeper drawdown -> higher stress
    comp = (z_vix + z_hy + z_curve + z_dd) / 4.0
    out = df[["date"]].copy()
    out["stress_0_100"] = (comp * 100).round(1)
    import altair as alt
    ch = alt.Chart(out).mark_line().encode(x="date:T", y=alt.Y("stress_0_100:Q", title="Market Stress (0-100)"))
    st.altair_chart(ch, use_container_width=True)
    st.caption("Composite blends VIX, HY OAS (FRED), 10y–2y curve (FRED), and SPY drawdown into a 0–100 scale.")

# Router for Market Analytics
# Replace the whole function
def show_market_analytics():
    # Shared controls for BOTH views
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        ticker = st.text_input("Ticker (Yahoo Finance)", value="SPY", key="ma_ticker")
    with c2:
        start_year = st.number_input("Start Year", min_value=1900, max_value=datetime.now().year, value=2020, key="ma_start")
    with c3:
        k = st.slider("Similar years (for Market Memory)", min_value=3, max_value=10, value=5, key="ma_k")

    if not ticker:
        return

    # Render both views on the same page with the same security
    view_monthly_seasonality(ticker_override=ticker, start_year_override=start_year)
    st.markdown("---")
    view_market_memory(ticker_override=ticker, start_year_override=start_year, k_override=k)


# ---- Main ----

def main() -> None:
    st.set_page_config(page_title="Fund Monitoring Dashboard", layout="wide")
    with st.sidebar:
        page = option_menu(
            "Navigation",
            ["Fund Database", "Performance Est", "Market Views", "Fund Monitor", "Market Analytics"],  # :contentReference[oaicite:1]{index=1}
            default_index=0,
            orientation="vertical"
        )
    if page == "Fund Database":  # :contentReference[oaicite:2]{index=2}
        st.header("Fund Database")
        show_fund_database()
    
    elif page == "Performance Est":
        st.header("Performance Estimates")
        st.write("Performance estimates are sent by hedge funds to investment.coverage@brightside-capital.com.")
        st.write("Automation and data extraction from emails happens via n8n and ChatGPT API.")
        st.write("Data stored in a cloud drive and pulled in below.")
        show_performance_view()

    elif page == "Market Views":
        st.write("## Fund Positions and Investment Thesis")
        st.write("Latest manager portfolio positions are extracted from fund letters, factsheets using LLMs and investment thesis performance are tracked.")
        st.write("Automatic PDF extraction when files are received via n8n workflowand ChatGPT API.")
        st.write("Data stored in a cloud drive and pulled in/transformed below.")
        show_market_view()
    elif page == "Fund Monitor":
        st.header("Fund Monitor")
        st.write("Fund data received via email attachments (pdf) are stored in a cloud drive.")
        st.write("Exposures, historical returns and other metrics are extracted using the ChatGPT API inside the n8n workflow.")
        st.write("Data stored in a cloud drive and pulled in/transformed below.")
        show_fund_monitor()
    elif page == "Market Analytics":
        st.header("Market Analytics")
        st.write("The idea here is to build a collection of market signals and indicators that provide the AI model with current market context i.e. what is going on? in order to connext the dots.")
        show_market_analytics()

if __name__ == "__main__":
    main()
