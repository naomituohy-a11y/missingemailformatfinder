import os
import re
import tempfile
from collections import defaultdict, Counter

import pandas as pd
import streamlit as st
from unidecode import unidecode

# =========================
# Config / Paths
# =========================
REPO_PATH = os.environ.get("REPO_PATH", "./master_repository.parquet")  # set to /data/master_repository.parquet on Railway

# =========================
# Column detection helpers
# =========================
def match_col(possible_names, df_cols):
    low_map = {c.lower(): c for c in df_cols}
    for name in possible_names:
        if name.lower() in low_map:
            return low_map[name.lower()]
    return None

ALIASES_COMPANY = ["Company","Company Name","Account","Account Name","Organisation","Organization","Company Name for Emails"]
ALIASES_EMAIL   = ["Email","Email Address","Primary Email","Work Email","Business Email","E-mail","Email(2)"]
ALIASES_FIRST   = ["First Name","Firstname","Given Name","Forename","First"]
ALIASES_LAST    = ["Last Name","Lastname","Surname","Family Name","Last"]
ALIASES_DOMAIN  = ["Domain","Company Domain","Website","Company Website","Company Domain/Website","Domain(2)"]  # optional
ALIASES_COUNTRY = ["Country","Company Country","Office Country","HQ Country","Country/Region"]                   # optional

SEPARATORS = ["", ".", "_", "-"]
PATTERN_KEYS = ["first.last","f.lastname","firstname.l","f.l","firstlast","flast","lastfirst","last.f","first"]

COUNTRY_TLD_PREFS = {
    "ireland":[".ie",".com"],
    "united kingdom":[".co.uk",".uk",".com"], "uk":[".co.uk",".uk",".com"], "england":[".co.uk",".uk",".com"],
    "scotland":[".co.uk",".uk",".com"], "wales":[".co.uk",".uk",".com"],
    "germany":[".de",".com"], "france":[".fr",".com"], "spain":[".es",".com"], "italy":[".it",".com"],
    "netherlands":[".nl",".com"], "belgium":[".be",".com"], "sweden":[".se",".com"], "norway":[".no",".com"],
    "denmark":[".dk",".com"], "finland":[".fi",".com"], "poland":[".pl",".com"], "portugal":[".pt",".com"],
    "austria":[".at",".com"], "switzerland":[".ch",".com"], "czech republic":[".cz",".com"], "czechia":[".cz",".com"],
    "slovakia":[".sk",".com"], "hungary":[".hu",".com"], "romania":[".ro",".com"], "bulgaria":[".bg",".com"],
    "greece":[".gr",".com"], "turkey":[".com.tr",".tr",".com"],
    "united states":[".com",".us"], "usa":[".com",".us"], "canada":[".ca",".com"],
    "australia":[".com.au",".au",".com"], "new zealand":[".co.nz",".nz",".com"]
}

def clean_string(x):
    if pd.isna(x): return pd.NA
    s = str(x).strip()
    return s if s else pd.NA

def normalize_country(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    s = str(x).strip().lower()
    return s or None

def norm_name(x):
    if pd.isna(x): return ""
    x = unidecode(str(x)).lower().strip()
    x = re.sub(r"[^\w\s'-]", "", x)
    x = x.replace("’", "'")
    base = re.sub(r"[\s'-]+", "", x)
    return base

def extract_domain(email):
    if not isinstance(email, str): return None
    m = re.search(r"@([\w\.-]+\.[A-Za-z]{2,})$", email.strip())
    return m.group(1).lower() if m else None

def local_part(email):
    if not isinstance(email, str) or "@" not in email: return None
    return email.split("@",1)[0].lower().strip()

def pattern_candidates(first, last):
    f = norm_name(first); l = norm_name(last)
    fi = f[:1]; li = l[:1]
    cands = []
    for sep in SEPARATORS:
        cands.extend([
            ("first.last", f"{f}{sep}{l}"),
            ("f.lastname", f"{fi}{sep}{l}"),
            ("firstname.l", f"{f}{sep}{li}"),
            ("f.l",        f"{fi}{sep}{li}"),
            ("firstlast",  f"{f}{l}"),
            ("flast",      f"{fi}{l}"),
            ("lastfirst",  f"{l}{f}"),
            ("last.f",     f"{l}{sep}{fi}"),
            ("first",      f"{f}"),
        ])
    seen, uniq = set(), []
    for k,v in cands:
        if (k,v) not in seen:
            seen.add((k,v))
            uniq.append((k,v))
    return uniq

def detect_email_pattern(first, last, email):
    lp = local_part(email)
    if not lp: return None
    for key, cand in pattern_candidates(first, last):
        if lp == cand:
            return key
    return None

def generate_email(first, last, pattern_key, domain):
    f = norm_name(first); l = norm_name(last)
    fi = f[:1]; li = l[:1]
    def with_sep(a,b):
        for sep in [".","_","-"]:
            return f"{a}{sep}{b}@{domain}"
        return f"{a}{b}@{domain}"
    if pattern_key == "first.last": return with_sep(f,l)
    if pattern_key == "f.lastname": return with_sep(fi,l)
    if pattern_key == "firstname.l":return with_sep(f,li)
    if pattern_key == "f.l":        return with_sep(fi,li)
    if pattern_key == "firstlast":  return f"{f}{l}@{domain}"
    if pattern_key == "flast":      return f"{fi}{l}@{domain}"
    if pattern_key == "lastfirst":  return f"{l}{f}@{domain}"
    if pattern_key == "last.f":     return with_sep(l,fi)
    if pattern_key == "first":      return f"{f}@{domain}"
    return None

# =========================
# Column normalization
# =========================
def ensure_cols(df):
    c_company = match_col(ALIASES_COMPANY, df.columns)
    c_email   = match_col(ALIASES_EMAIL, df.columns)
    c_first   = match_col(ALIASES_FIRST, df.columns)
    c_last    = match_col(ALIASES_LAST, df.columns)
    c_domain  = match_col(ALIASES_DOMAIN, df.columns)   # optional
    c_country = match_col(ALIASES_COUNTRY, df.columns)  # optional

    # fallback from Full Name
    if c_first is None or c_last is None:
        name_col = match_col(["Name","Full Name"], df.columns)
        if name_col:
            names = df[name_col].fillna("").astype(str)
            first_guess = names.str.split().str[0]
            last_guess  = names.str.split().str[-1]
            if c_first is None:
                df["First Name"] = first_guess; c_first = "First Name"
            if c_last is None:
                df["Last Name"]  = last_guess;  c_last  = "Last Name"

    missing = []
    if c_company is None: missing.append("Company")
    if c_email   is None: missing.append("Email")
    if c_first   is None: missing.append("First Name")
    if c_last    is None: missing.append("Last Name")
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.rename(columns={
        c_company: "Company",
        c_email:   "Email",
        c_first:   "First Name",
        c_last:    "Last Name",
        **({c_domain:"Domain"}  if c_domain  else {}),
        **({c_country:"Country"} if c_country else {}),
    })

    df["Company"]    = df["Company"].apply(clean_string)
    df["Email"]      = df["Email"].replace(r"^\s*$", pd.NA, regex=True)
    df["Email"]      = df["Email"].replace(["nan","NaN","None","NULL","null"], pd.NA)
    df["First Name"] = df["First Name"].apply(clean_string)
    df["Last Name"]  = df["Last Name"].apply(clean_string)

    if "Domain" in df.columns:
        df["Domain"] = df["Domain"].astype(str).str.strip()
        df["Domain"] = df["Domain"].replace(r"^\s*$", pd.NA, regex=True)
        df["Domain"] = df["Domain"].apply(lambda x: re.sub(r"^https?://","",x) if isinstance(x,str) else x)
        df["Domain"] = df["Domain"].apply(lambda x: x.split("/")[0] if isinstance(x,str) else x)
        df["Domain"] = df["Domain"].str.lower()

    if "Country" in df.columns:
        df["Country"] = df["Country"].apply(normalize_country)

    return df

# =========================
# Repo utilities (persistent)
# =========================
REPO_SCHEMA = ["company","country_norm","domain","pattern","count"]

def load_repo():
    if os.path.exists(REPO_PATH):
        return pd.read_parquet(REPO_PATH)
    return pd.DataFrame(columns=REPO_SCHEMA)

def save_repo(df_repo):
    df_repo = df_repo.copy()
    if "count" in df_repo.columns:
        df_repo["count"] = df_repo["count"].astype("int64")
    df_repo.to_parquet(REPO_PATH, compression="snappy")

def add_examples_to_repo(df_input):
    """Extract rows WITH emails → (company, country, domain, pattern, count=1), merge into persistent repo."""
    repo = load_repo()
    df = ensure_cols(df_input)

    rows = []
    for _, r in df.dropna(subset=["Email"]).iterrows():
        company = (r["Company"] or "")
        company_key = unidecode(str(company)).strip().lower()
        country_norm = normalize_country(r["Country"]) if "Country" in df.columns else None
        first, last, email = r["First Name"], r["Last Name"], str(r["Email"]).strip()
        dom = extract_domain(email)
        pat = detect_email_pattern(first, last, email)
        if company_key and dom and pat:
            rows.append([company_key, country_norm, dom, pat, 1])

    if not rows:
        return load_repo()

    new = pd.DataFrame(rows, columns=REPO_SCHEMA)
    combined = pd.concat([repo, new], ignore_index=True)
    combined = (combined.groupby(["company","country_norm","domain","pattern"], as_index=False)
                        .agg(count=("count","sum")))
    save_repo(combined)
    return combined

def choose_best_domain(domains, country_norm):
    if not domains: return None
    cnt = Counter(domains)
    if country_norm and country_norm in COUNTRY_TLD_PREFS:
        for pref in COUNTRY_TLD_PREFS[country_norm]:
            cands = [d for d in cnt if d.endswith(pref)]
            if cands:
                return sorted(cands, key=lambda d: (-cnt[d], d))[0]
    coms = [d for d in cnt if d.endswith(".com")]
    if coms:
        return sorted(coms, key=lambda d: (-cnt[d], d))[0]
    return sorted(cnt.keys(), key=lambda d: (-cnt[d], d))[0]

def build_maps_from_repo(df_repo):
    """Build maps for fast lookup."""
    repo_by_domain = defaultdict(lambda: defaultdict(int))
    repo_cc = defaultdict(lambda: defaultdict(list))
    repo_c  = defaultdict(lambda: defaultdict(list))
    for _, r in df_repo.iterrows():
        company = r["company"]; country_norm = r["country_norm"]
        dom = r["domain"]; pat = r["pattern"]; count = int(r["count"])
        repo_by_domain[dom][pat] += count
        for _i in range(count):
            repo_c[company][pat].append(dom)
            if country_norm:
                key = f"{company}|{country_norm}"
                repo_cc[key][pat].append(dom)
    return repo_by_domain, repo_cc, repo_c

def learn_from_dataset(df):
    df = ensure_cols(df)
    cc = defaultdict(lambda: defaultdict(list))
    c  = defaultdict(lambda: defaultdict(list))
    for _, r in df.dropna(subset=["Email"]).iterrows():
        company = unidecode(str(r["Company"])).strip().lower()
        country_norm = normalize_country(r["Country"]) if "Country" in df.columns else None
        first, last, email = r["First Name"], r["Last Name"], str(r["Email"]).strip()
        dom = extract_domain(email); pat = detect_email_pattern(first, last, email)
        if company and dom and pat:
            c[company][pat].append(dom)
            if country_norm:
                cc[f"{company}|{country_norm}"][pat].append(dom)
    return cc, c

def best_from_sources(company_key, country_norm, repo_cc, repo_c, cc_data, c_data):
    # company+country → repo then data
    if country_norm:
        key = f"{company_key}|{country_norm}"
        for src_tag, src in (("REPO company+country", repo_cc), ("DATA company+country", cc_data)):
            if key in src and src[key]:
                fmt_counts = {k: len(v) for k,v in src[key].items() if v}
                if fmt_counts:
                    pat = max(fmt_counts, key=fmt_counts.get)
                    best_dom = choose_best_domain(src[key][pat], country_norm)
                    if best_dom: return pat, best_dom, src_tag
    # company only → repo then data
    for src_tag, src in (("REPO company", repo_c), ("DATA company", c_data)):
        if company_key in src and src[company_key]:
            fmt_counts = {k: len(v) for k,v in src[company_key].items() if v}
            if fmt_counts:
                pat = max(fmt_counts, key=fmt_counts.get)
                best_dom = choose_best_domain(src[company_key][pat], country_norm)
                if best_dom: return pat, best_dom, src_tag
    return None, None, None

def fill_missing_emails(df, learn_from_this_file=False):
    df = ensure_cols(df)

    # audit columns
    for col in ["Email_Filled","Email_Fill_Reason","Email_Pattern_Detected","Domain_Guess"]:
        if col not in df.columns: df[col] = pd.NA
    df["Email_Filled"] = df["Email_Filled"].fillna("No")

    # learn from this file first (incremental)
    if learn_from_this_file:
        add_examples_to_repo(df)

    repo_df = load_repo()
    _, repo_cc, repo_c = build_maps_from_repo(repo_df)
    cc_data, c_data = learn_from_dataset(df)

    # fill
    to_fill = df.index[df["Email"].isna()].tolist()
    for idx in to_fill:
        r = df.loc[idx]
        company_key = unidecode(str(r["Company"])).strip().lower() if pd.notna(r["Company"]) else ""
        if not company_key: continue
        first, last = r["First Name"], r["Last Name"]
        if pd.isna(first) or pd.isna(last): continue
        country_norm = normalize_country(r["Country"]) if "Country" in df.columns else None

        pat, domain, src_tag = best_from_sources(company_key, country_norm, repo_cc, repo_c, cc_data, c_data)
        if pat and domain:
            email = generate_email(first, last, pat, domain)
            if email:
                df.at[idx,"Email"] = email
                df.at[idx,"Email_Filled"] = "Yes"
                df.at[idx,"Email_Fill_Reason"] = f"Filled via {src_tag} (pattern '{pat}', domain '{domain}')"
                df.at[idx,"Domain_Guess"] = domain

    # audit: existing emails → pattern + domain guess
    detected = []
    for i, r in df.iterrows():
        if pd.isna(r["Email"]):
            detected.append(pd.NA)
        else:
            detected.append(detect_email_pattern(r["First Name"], r["Last Name"], r["Email"]))
            dom = extract_domain(r["Email"])
            if dom and pd.isna(r.get("Domain_Guess", pd.NA)):
                df.at[i,"Domain_Guess"] = dom
    df["Email_Pattern_Detected"] = detected

    return df

# =========================
# File loader
# =========================
def load_df(uploaded_file):
    if uploaded_file is None: return None
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()
    if ext in [".xlsx",".xlsm",".xltx",".xltm",".xls"]:
        return pd.read_excel(uploaded_file)
    if ext == ".csv":
        return pd.read_csv(uploaded_file, keep_default_na=True, low_memory=False)
    if ext in [".parquet",".pq"]:
        return pd.read_parquet(uploaded_file)
    st.error(f"Unsupported file type: {ext}")
    return None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Email Format Filler", layout="wide")
st.title("Email Format Filler (Persistent Repository)")

tab_init, tab_fill = st.tabs(["Initialize / Replace Master", "Fill Missing Emails"])

with tab_init:
    st.markdown("Upload your large master **once**. We’ll parse what we need and save a compact repository to persistent storage.")
    master = st.file_uploader("Master Repository (CSV/XLSX/Parquet)", type=["csv","xlsx","xls","parquet","pq"])
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Build / Replace Persistent Repository", type="primary", use_container_width=True):
            if not master:
                st.error("Please upload a Master Repository file.")
            else:
                df = load_df(master)
                repo = add_examples_to_repo(df)
                save_repo(repo)
                st.success("Repository built and saved.")
                st.write({
                    "unique_keys": int(repo.shape[0]),
                    "total_examples": int(repo["count"].sum()) if not repo.empty else 0
                })
    with colB:
        if st.button("Download Current Repository", use_container_width=True):
            repo = load_repo()
            if repo.empty:
                st.warning("Repository is empty.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                repo.to_parquet(tmp.name, compression="snappy")
                with open(tmp.name, "rb") as f:
                    st.download_button("Download .parquet", f, file_name="master_repository.parquet", mime="application/octet-stream")

with tab_fill:
    st.markdown("Upload a working Apollo-style file (blank emails allowed). Optionally **learn** from known emails in this file, then fill the blanks.")
    work = st.file_uploader("Working Dataset (CSV/XLSX)", type=["csv","xlsx","xls"], key="work")
    learn = st.checkbox("Learn from this file (add existing emails to the repository)", value=True)
    out_fmt = st.radio("Output format", ["Excel (.xlsx)","CSV (.csv)"], horizontal=True)

    if st.button("Run Fill", type="primary"):
        if not work:
            st.error("Please upload a Working Dataset.")
        else:
            df = load_df(work)
            result = fill_missing_emails(df, learn_from_this_file=learn)
            st.success("Done.")
            st.dataframe(result.head(100), use_container_width=True)

            if out_fmt.startswith("Excel"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                result.to_excel(tmp.name, index=False)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download result (.xlsx)", f, file_name="emails_filled.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                result.to_csv(tmp.name, index=False)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download result (.csv)", f, file_name="emails_filled.csv", mime="text/csv")
