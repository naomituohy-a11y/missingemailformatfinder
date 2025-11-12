# main.py — Email Format Filler (Persistent Repository, no repo-upload UI)

import os, re, io
from collections import defaultdict, Counter

import pandas as pd
import streamlit as st
from unidecode import unidecode

st.set_page_config(page_title="Email Format Filler", layout="wide")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
# Where to look for the compact repository parquet.
# You can override with Railway env var REPO_PATH.
REPO_PATH = os.environ.get("REPO_PATH", "/data/master_repository.parquet")
REPO_SCHEMA = ["company", "country_norm", "domain", "pattern", "count"]

CANDIDATE_PATHS = [
    REPO_PATH,
    "./master_repository.parquet",
    "./master_repository_compact.parquet",
    "/data/master_repository_compact.parquet",
]

# ------------------------------------------------------------------
# NORMALIZERS (shared with repo build)
# ------------------------------------------------------------------
LEGAL_STOPWORDS = {
    "the","group","holding","holdings","international","company","co","inc","incorporated",
    "limited","ltd","plc","llc","llp","gmbh","ag","sa","spa","srl","bv","nv","as","ab","oy","aps",
    "kft","zrt","rt","sarl","sas","pte","pty","bhd","sdn","kk","dmcc","pjsc","jsc","ltda",
    "corp","corporation","co."
}

def normalize_company_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    s = unidecode(name).lower()
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    toks = [t for t in re.split(r"[\s-]+", s) if t]
    toks = [t for t in toks if t not in LEGAL_STOPWORDS]
    return "".join(toks)

def normalize_country(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = unidecode(str(x)).strip().lower()
    if not s:
        return None
    alias = {
        "u.s.": "united states",
        "usa": "united states",
        "uk": "united kingdom",
    }
    return alias.get(s, s)

def norm_name(x):
    if pd.isna(x): return ""
    x = unidecode(str(x)).lower().strip()
    x = re.sub(r"[^\w\s'-]", "", x)
    x = x.replace("’","'")
    return re.sub(r"[\s'-]+","",x)

# ------------------------------------------------------------------
# REPOSITORY LOADING & MAPS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_repo_df():
    last_err = None
    for p in CANDIDATE_PATHS:
        try:
            if os.path.exists(p):
                df = pd.read_parquet(p)
                if sorted(df.columns.tolist()) != sorted(REPO_SCHEMA):
                    raise ValueError(
                        f"Wrong columns in {p}. Found {df.columns.tolist()}, "
                        f"expected {REPO_SCHEMA}."
                    )
                return df, p
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Unable to load a valid repository from {CANDIDATE_PATHS}. "
        f"Last error: {last_err}"
    )

def build_repo_maps(repo_df: pd.DataFrame):
    cc = defaultdict(Counter)  # (company,country) -> Counter[(pattern,domain)]
    c  = defaultdict(Counter)  # company -> Counter[(pattern,domain)]
    for company, ctry, dom, pat, cnt in repo_df[["company","country_norm","domain","pattern","count"]].itertuples(index=False):
        cnt = int(cnt) if pd.notna(cnt) else 1
        cc[(company, ctry)][(pat, dom)] += cnt
        c[company][(pat, dom)] += cnt
    return cc, c

def repository_status_box(repo_df: pd.DataFrame, repo_path: str):
    with st.expander("📦 Repository status", expanded=True):
        st.write({
            "path": repo_path,
            "rows": int(len(repo_df)),
            "unique_companies": int(repo_df["company"].nunique()),
            "columns": repo_df.columns.tolist()
        })
        st.caption("Repository must have schema: ['company','country_norm','domain','pattern','count'].")

# ------------------------------------------------------------------
# INPUT COLUMN DETECTION
# ------------------------------------------------------------------
def find_col(df, options):
    low = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in low:
            return low[o.lower()]
    return None

def detect_columns(df):
    c_company = find_col(df, ["Company","Company Name","Company Name for Emails","Account","Organisation"])
    c_email   = find_col(df, ["Email","Primary Email","Work Email","Business Email","E-mail"])
    c_first   = find_col(df, ["First Name","Firstname","Given Name","Forename"])
    c_last    = find_col(df, ["Last Name","Lastname","Surname","Family Name"])
    c_country = find_col(df, ["Country","Company Country","Office Country","HQ Country","Country/Region"])
    return c_company, c_email, c_first, c_last, c_country

# ------------------------------------------------------------------
# EMAIL GENERATION
# ------------------------------------------------------------------
def generate_email(first, last, fmt, domain):
    if pd.isna(first) or pd.isna(last) or not domain:
        return None
    f = norm_name(first); l = norm_name(last)
    if not f or not l:
        return None
    fi, li = f[:1], l[:1]
    if   fmt == 'first.last':   return f"{f}.{l}@{domain}"
    elif fmt == 'f.lastname':   return f"{fi}.{l}@{domain}"
    elif fmt == 'firstname.l':  return f"{f}.{li}@{domain}"
    elif fmt == 'f.l':          return f"{fi}.{li}@{domain}"
    elif fmt == 'firstlast':    return f"{f}{l}@{domain}"
    elif fmt == 'flast':        return f"{fi}{l}@{domain}"
    elif fmt == 'lastfirst':    return f"{l}{f}@{domain}"
    elif fmt == 'last.f':       return f"{l}.{fi}@{domain}"
    elif fmt == 'first':        return f"{f}@{domain}"
    return None

# ------------------------------------------------------------------
# PRE-FLIGHT REPO REACH
# ------------------------------------------------------------------
def calc_repo_reach(df, c_company, c_country, repo_cc, repo_c):
    n_total = int(len(df))
    if not c_company:
        return {"rows_in_file": n_total, "rows_with_company": 0,
                "repo_hits_company_country": 0, "repo_hits_company_only": 0, "repo_misses": n_total}

    comp_keys = df[c_company].fillna("").map(normalize_company_key)
    cnorms    = df[c_country].map(normalize_country) if c_country else pd.Series([None]*len(df))

    hits_cc = 0; hits_c = 0
    for ck, ct in zip(comp_keys, cnorms):
        if ck:
            if repo_cc.get((ck, ct)):
                hits_cc += 1
            elif repo_c.get(ck):
                hits_c += 1
    return {
        "rows_in_file": n_total,
        "rows_with_company": int(df[c_company].notna().sum()) if c_company else 0,
        "repo_hits_company_country": hits_cc,
        "repo_hits_company_only": hits_c,
        "repo_misses": n_total - (hits_cc + hits_c)
    }

# ------------------------------------------------------------------
# FILL (repo-first)
# ------------------------------------------------------------------
def fill_missing_emails(df, c_company, c_email, c_first, c_last, c_country, repo_cc, repo_c):
    out = df.copy()
    filled_rows = 0
    src_repo_cc = 0
    src_repo_c  = 0

    if not all([c_company, c_email, c_first, c_last]):
        raise ValueError("Missing required columns — need Company, Email, First Name, Last Name.")

    mask_missing = out[c_email].isna() | (out[c_email].astype(str).str.strip().eq(""))
    rows = out[mask_missing].index.tolist()

    prog = st.progress(0)
    status = st.empty()

    total = len(rows) if rows else 1
    for i, idx in enumerate(rows, 1):
        row = out.loc[idx]
        ckey = normalize_company_key(row[c_company])
        cnrm = normalize_country(row[c_country]) if c_country else None
        first, last = row[c_first], row[c_last]

        new_email = None

        # repo (company+country)
        hit_cc = repo_cc.get((ckey, cnrm))
        if hit_cc:
            (pat, dom), _ = max(hit_cc.items(), key=lambda kv: kv[1])
            new_email = generate_email(first, last, pat, dom)
            if new_email:
                src_repo_cc += 1

        # repo (company only)
        if new_email is None:
            hit_c = repo_c.get(ckey)
            if hit_c:
                (pat, dom), _ = max(hit_c.items(), key=lambda kv: kv[1])
                new_email = generate_email(first, last, pat, dom)
                if new_email:
                    src_repo_c += 1

        if new_email:
            out.at[idx, c_email] = new_email
            filled_rows += 1

        if i % 25 == 0 or i == total:
            prog.progress(int(i/total*100))
            status.text(f"Processed {i}/{total} missing rows … Filled so far: {filled_rows}")

    prog.progress(100)
    status.empty()
    return out, {"filled": filled_rows, "src_repo_cc": src_repo_cc, "src_repo_c": src_repo_c, "considered": len(rows)}

# ------------------------------------------------------------------
# UI (no repo upload; repo must exist in app image)
# ------------------------------------------------------------------
st.title("Email Format Filler — Persistent Repository (No-Upload)")

# Load repository or stop with a clear error
try:
    repo_df, repo_path = load_repo_df()
except Exception as e:
    st.error(f"Repository load failed: {e}")
    st.stop()

repository_status_box(repo_df, repo_path)
repo_cc, repo_c = build_repo_maps(repo_df)

st.write("---")
st.header("Fill Missing Emails")

uploaded = st.file_uploader("Upload your CSV/XLSX (Apollo-style export).", type=["csv","xlsx","xls"])
if not uploaded:
    st.caption("Tip: the file must include Company, First Name, Last Name, and an Email column (blank where missing).")
    st.stop()

# Load input file
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded, keep_default_na=True, low_memory=False)
else:
    df = pd.read_excel(uploaded, engine="openpyxl")

st.success(f"Loaded file: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Detect columns
c_company, c_email, c_first, c_last, c_country = detect_columns(df)
st.write("**Detected columns:**", {
    "Company": c_company, "Email": c_email, "First Name": c_first,
    "Last Name": c_last, "Country": c_country
})

# Pre-flight reach
stats = calc_repo_reach(df, c_company, c_country, repo_cc, repo_c)
st.info({
    "rows_in_file": stats["rows_in_file"],
    "rows_with_company": stats["rows_with_company"],
    "repo_hits_company_country (pre-flight)": stats["repo_hits_company_country"],
    "repo_hits_company_only (pre-flight)": stats["repo_hits_company_only"],
    "repo_misses_pre_flight": stats["repo_misses"]
})

# Misses preview (first 15)
with st.expander("🔍 Show first 15 repo misses (normalized keys)"):
    misses = []
    if c_company:
        comp_keys = df[c_company].fillna("").map(normalize_company_key)
        cnorms    = df[c_country].map(normalize_country) if c_country else pd.Series([None]*len(df))
        for i, (ck, ct) in enumerate(zip(comp_keys, cnorms)):
            if not ck:
                continue
            if not repo_cc.get((ck, ct)) and not repo_c.get(ck):
                misses.append({"row_index": i, "company_norm": ck, "country_norm": ct})
                if len(misses) >= 15:
                    break
    if misses:
        st.dataframe(pd.DataFrame(misses))
    else:
        st.write("No misses found in the inspected sample.")

# Live lookup tester
with st.expander("🧪 Test a repository lookup"):
    comp_in = st.text_input("Company (exactly as it appears in your file)")
    country_in = st.text_input("Country (optional)")
    if st.button("Lookup"):
        ck = normalize_company_key(comp_in)
        ct = normalize_country(country_in) if country_in else None
        cc_hit = repo_cc.get((ck, ct))
        c_hit  = repo_c.get(ck)
        if cc_hit:
            (pat, dom), cnt = max(cc_hit.items(), key=lambda kv: kv[1])
            st.success(f"Match by (company,country): pattern={pat}, domain={dom}, weight={cnt}")
        elif c_hit:
            (pat, dom), cnt = max(c_hit.items(), key=lambda kv: kv[1])
            st.warning(f"Match by company only: pattern={pat}, domain={dom}, weight={cnt}")
        else:
            st.error(f"No repo match. normalized_company='{ck}', normalized_country='{ct}'")

st.write("---")
if st.button("▶️ Run Fill (Repo-first)"):
    result_df, fill_stats = fill_missing_emails(
        df, c_company, c_email, c_first, c_last, c_country, repo_cc, repo_c
    )
    filled, considered = fill_stats["filled"], fill_stats["considered"]
    st.success(
        f"Filled {filled:,} of {considered:,} missing emails "
        f"({(filled/considered*100 if considered else 0):.1f}%)."
    )
    st.write({
        "from_repo_company_country": fill_stats["src_repo_cc"],
        "from_repo_company_only": fill_stats["src_repo_c"],
    })

    # Download (CSV)
    buf = io.BytesIO()
    result_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download CSV", buf.getvalue(), file_name="emails_filled.csv", mime="text/csv")
