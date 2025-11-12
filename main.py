# main.py — Email Format Filler
# Hybrid strategy: (1) infer from UPLOADED FILE, then (2) fall back to PERSISTENT REPOSITORY

import os, re, io
from collections import defaultdict, Counter

import pandas as pd
import streamlit as st
from unidecode import unidecode

st.set_page_config(page_title="Email Format Filler — Hybrid", layout="wide")

# -----------------------------
# Config / Paths
# -----------------------------
CANDIDATE_REPO_PATHS = [
    os.environ.get("REPO_PATH", ""),  # allow override
    "./master_repository.parquet",
    "./master_repository_compact.parquet",
    "/data/master_repository.parquet",
    "/data/master_repository_compact.parquet",
]
REPO_SCHEMA = ["company", "country_norm", "domain", "pattern", "count"]

LEGAL_STOPWORDS = {
    "the","group","holding","holdings","international","company","co","inc","incorporated",
    "limited","ltd","plc","llc","llp","gmbh","ag","sa","spa","srl","bv","nv","as","ab","oy","aps",
    "kft","zrt","rt","sarl","sas","pte","pty","bhd","sdn","kk","dmcc","pjsc","jsc","ltda",
    "corp","corporation","co."
}

# -----------------------------
# Normalizers
# -----------------------------
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
        "u.k.": "united kingdom",
        "uk": "united kingdom",
        "england": "united kingdom",
        "gb": "united kingdom",
        "great britain": "united kingdom",
    }
    return alias.get(s, s)

def norm_name(x):
    if pd.isna(x): return ""
    x = unidecode(str(x)).lower().strip()
    x = re.sub(r"[^\w\s'-]", "", x)
    x = x.replace("’","'")
    return re.sub(r"[\s'-]+","",x)

# -----------------------------
# Pattern detection on a known email + names
# -----------------------------
def detect_email_pattern(first, last, email):
    """Return canonical pattern string or None based on first/last + email."""
    if not isinstance(email, str) or "@" not in email:
        return None
    local = email.split("@")[0].lower()

    f = norm_name(first)
    l = norm_name(last)
    if not f or not l:
        return None

    cand = [
        ("first.last", f"{f}.{l}"),
        ("f.lastname", f"{f[:1]}.{l}"),
        ("firstname.l", f"{f}.{l[:1]}"),
        ("f.l", f"{f[:1]}.{l[:1]}"),
        ("firstlast", f"{f}{l}"),
        ("flast", f"{f[:1]}{l}"),
        ("lastfirst", f"{l}{f}"),
        ("last.f", f"{l}.{f[:1]}"),
        ("first", f"{f}"),
    ]
    for pat, target in cand:
        if local == target:
            return pat
    return None

# -----------------------------
# Generate from pattern
# -----------------------------
def generate_email(first, last, fmt, domain):
    if pd.isna(first) or pd.isna(last) or not domain:
        return None
    f = norm_name(first); l = norm_name(last)
    if not f or not l:
        return None
    fi, li = f[:1], l[:1]
    if   fmt == 'first.last':   local = f"{f}.{l}"
    elif fmt == 'f.lastname':   local = f"{fi}.{l}"
    elif fmt == 'firstname.l':  local = f"{f}.{li}"
    elif fmt == 'f.l':          local = f"{fi}.{li}"
    elif fmt == 'firstlast':    local = f"{f}{l}"
    elif fmt == 'flast':        local = f"{fi}{l}"
    elif fmt == 'lastfirst':    local = f"{l}{f}"
    elif fmt == 'last.f':       local = f"{l}.{fi}"
    elif fmt == 'first':        local = f"{f}"
    else: return None
    return f"{local}@{domain}"

# -----------------------------
# Repo loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_repo_df():
    last_err = None
    for p in [p for p in CANDIDATE_REPO_PATHS if p]:
        try:
            if os.path.exists(p):
                df = pd.read_parquet(p)
                if sorted(df.columns.tolist()) != sorted(REPO_SCHEMA):
                    raise ValueError(
                        f"Wrong columns in {p}. Found {df.columns.tolist()}, expected {REPO_SCHEMA}."
                    )
                return df, p
        except Exception as e:
            last_err = e
    return None, f"Unable to load a valid repository from {CANDIDATE_REPO_PATHS}. Last error: {last_err}"

def build_maps_from_repo(repo_df: pd.DataFrame):
    cc = defaultdict(Counter)  # (company,country) -> Counter[(pattern,domain)]
    c  = defaultdict(Counter)  # company -> Counter[(pattern,domain)]
    for company, ctry, dom, pat, cnt in repo_df[["company","country_norm","domain","pattern","count"]].itertuples(index=False):
        cnt = int(cnt) if pd.notna(cnt) else 1
        cc[(company, ctry)][(pat, dom)] += cnt
        c[company][(pat, dom)] += cnt
    return cc, c

# -----------------------------
# Build maps from the uploaded file (LOCAL inference)
# -----------------------------
def build_maps_from_local(df, c_company, c_email, c_first, c_last, c_country):
    cc_local = defaultdict(Counter)
    c_local  = defaultdict(Counter)

    if not all([c_company, c_email, c_first, c_last]):
        return cc_local, c_local

    present = df[(df[c_email].notna()) & (df[c_email].astype(str).str.contains("@"))]
    for _, row in present.iterrows():
        comp = normalize_company_key(row[c_company])
        if not comp:
            continue
        email = str(row[c_email]).strip().lower()
        domain = email.split("@")[1] if "@" in email else None
        pat = detect_email_pattern(row[c_first], row[c_last], email)
        if not domain or not pat:
            continue
        ctry = normalize_country(row[c_country]) if c_country else None
        cc_local[(comp, ctry)][(pat, domain)] += 1
        c_local[comp][(pat, domain)] += 1
    return cc_local, c_local

# -----------------------------
# Helpers
# -----------------------------
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

def choose_best(counter: Counter):
    """Pick the most common (pattern,domain). If tied, prefer .com."""
    if not counter:
        return None
    items = counter.most_common()
    best, best_count = items[0]
    # tie-breaker: prefer .com domain if counts equal
    ties = [x for x in items if x[1] == best_count]
    if len(ties) > 1:
        for (pat, dom), _ in ties:
            if dom.endswith(".com"):
                return (pat, dom)
    return best

# -----------------------------
# Pre-flight reach
# -----------------------------
def calc_reach(df, c_company, c_country, cc_local, c_local, cc_repo, c_repo):
    n = len(df)
    if not c_company:
        return {"rows_in_file": n, "rows_with_company": 0,
                "local_hits_cc":0,"local_hits_c":0,"repo_hits_cc":0,"repo_hits_c":0}

    comp_keys = df[c_company].fillna("").map(normalize_company_key)
    cnorms    = df[c_country].map(normalize_country) if c_country else pd.Series([None]*len(df))

    loc_cc=loc_c=rep_cc=rep_c=0
    for ck, ct in zip(comp_keys, cnorms):
        if not ck: 
            continue
        if cc_local.get((ck, ct)): loc_cc += 1
        elif c_local.get(ck):      loc_c  += 1
        elif cc_repo.get((ck, ct)):rep_cc += 1
        elif c_repo.get(ck):       rep_c  += 1
    return {
        "rows_in_file": n,
        "rows_with_company": int(df[c_company].notna().sum()),
        "local_hits_cc": loc_cc,
        "local_hits_c": loc_c,
        "repo_hits_cc": rep_cc,
        "repo_hits_c": rep_c
    }

# -----------------------------
# Fill: LOCAL first, then REPO
# -----------------------------
def fill_hybrid(df, c_company, c_email, c_first, c_last, c_country, cc_local, c_local, cc_repo, c_repo):
    out = df.copy()
    filled = 0
    src_local_cc = src_local_c = src_repo_cc = src_repo_c = 0

    if not all([c_company, c_email, c_first, c_last]):
        raise ValueError("Missing required columns — need Company, Email, First Name, Last Name.")

    target_idx = out[(out[c_email].isna()) | (out[c_email].astype(str).str.strip().eq(""))].index.tolist()
    total = len(target_idx) if target_idx else 1

    prog = st.progress(0)
    msg  = st.empty()

    for i, idx in enumerate(target_idx, 1):
        row = out.loc[idx]
        ck = normalize_company_key(row[c_company])
        if not ck:
            continue
        ct = normalize_country(row[c_country]) if c_country else None
        first, last = row[c_first], row[c_last]
        new_email = None

        # 1) LOCAL (company + country)
        best = choose_best(cc_local.get((ck, ct), Counter()))
        if best:
            pat, dom = best
            new_email = generate_email(first, last, pat, dom)
            if new_email: src_local_cc += 1

        # 2) LOCAL (company only)
        if new_email is None:
            best = choose_best(c_local.get(ck, Counter()))
            if best:
                pat, dom = best
                new_email = generate_email(first, last, pat, dom)
                if new_email: src_local_c += 1

        # 3) REPO (company + country)
        if new_email is None:
            best = choose_best(cc_repo.get((ck, ct), Counter()))
            if best:
                pat, dom = best
                new_email = generate_email(first, last, pat, dom)
                if new_email: src_repo_cc += 1

        # 4) REPO (company only)
        if new_email is None:
            best = choose_best(c_repo.get(ck, Counter()))
            if best:
                pat, dom = best
                new_email = generate_email(first, last, pat, dom)
                if new_email: src_repo_c += 1

        if new_email:
            out.at[idx, c_email] = new_email
            filled += 1

        if i % 25 == 0 or i == total:
            prog.progress(int(i / total * 100))
            msg.text(f"Processed {i}/{total} missing rows • Filled so far: {filled}")

    prog.progress(100)
    msg.empty()
    return out, {
        "filled": filled,
        "considered": len(target_idx),
        "src_local_cc": src_local_cc,
        "src_local_c": src_local_c,
        "src_repo_cc": src_repo_cc,
        "src_repo_c": src_repo_c
    }

# -----------------------------
# UI
# -----------------------------
st.title("Email Format Filler — Persistent Repository (Hybrid: Local → Repo)")

# Load repository (optional, but recommended)
repo_df, repo_info = load_repo_df()
if repo_df is None:
    st.warning(repo_info)
else:
    with st.expander("📦 Repository status", expanded=True):
        st.write({
            "path": repo_info,
            "rows": int(len(repo_df)),
            "unique_companies": int(repo_df["company"].nunique()),
            "columns": repo_df.columns.tolist()
        })

st.write("---")
st.header("Fill Missing Emails")

upl = st.file_uploader("Upload your CSV/XLSX (Apollo-style export).", type=["csv","xlsx","xls"])
if not upl:
    st.caption("Must include: Company, First Name, Last Name, and Email (blank where missing). Country is optional but improves accuracy.")
    st.stop()

# Read input
if upl.name.lower().endswith(".csv"):
    df = pd.read_csv(upl, keep_default_na=True, low_memory=False)
else:
    df = pd.read_excel(upl, engine="openpyxl")

st.success(f"Loaded file: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

# Detect columns
c_company, c_email, c_first, c_last, c_country = detect_columns(df)
st.write("**Detected columns:**", {
    "Company": c_company, "Email": c_email, "First Name": c_first,
    "Last Name": c_last, "Country": c_country
})

# Build maps
cc_local, c_local = build_maps_from_local(df, c_company, c_email, c_first, c_last, c_country)
if repo_df is not None:
    cc_repo,  c_repo  = build_maps_from_repo(repo_df)
else:
    cc_repo, c_repo = defaultdict(Counter), defaultdict(Counter)

# Pre-flight
reach = calc_reach(df, c_company, c_country, cc_local, c_local, cc_repo, c_repo)
st.info({
    "rows_in_file": reach["rows_in_file"],
    "rows_with_company": reach["rows_with_company"],
    "local_hits_company_country": reach["local_hits_cc"],
    "local_hits_company_only": reach["local_hits_c"],
    "repo_hits_company_country": reach["repo_hits_cc"],
    "repo_hits_company_only": reach["repo_hits_c"],
})

# Optional: quick tester
with st.expander("🧪 Test lookup (shows what would be used)"):
    comp_in   = st.text_input("Company")
    country_in= st.text_input("Country (optional)")
    if st.button("Lookup"):
        ck = normalize_company_key(comp_in)
        ct = normalize_country(country_in) if country_in else None
        for label, source in [
            ("Local (company+country)", cc_local.get((ck, ct))),
            ("Local (company only)", c_local.get(ck)),
            ("Repo (company+country)", cc_repo.get((ck, ct))),
            ("Repo (company only)", c_repo.get(ck)),
        ]:
            if source:
                (pat, dom), cnt = choose_best(source), max(source.values()) if source else (None, 0)
                st.write(f"**{label}:** pattern={pat}, domain={dom}, weight≈{cnt}")
            else:
                st.write(f"**{label}:** none")

st.write("---")
if st.button("▶️ Run Fill"):
    result_df, stats = fill_hybrid(
        df, c_company, c_email, c_first, c_last, c_country,
        cc_local, c_local, cc_repo, c_repo
    )
    filled, considered = stats["filled"], stats["considered"]
    st.success(
        f"Filled {filled:,} of {considered:,} missing emails "
        f"({(filled/considered*100 if considered else 0):.1f}%)."
    )
    st.write({
        "from_local_company_country": stats["src_local_cc"],
        "from_local_company_only": stats["src_local_c"],
        "from_repo_company_country": stats["src_repo_cc"],
        "from_repo_company_only": stats["src_repo_c"],
    })

    # Download
    csv_buf = io.BytesIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download CSV", csv_buf.getvalue(), file_name="emails_filled.csv", mime="text/csv")
