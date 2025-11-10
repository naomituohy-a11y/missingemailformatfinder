import os
import re
import tempfile
from collections import defaultdict, Counter

import pandas as pd
import gradio as gr
from unidecode import unidecode

# =========================
# Config / Paths
# =========================
REPO_PATH = os.environ.get("REPO_PATH", "./master_repository.parquet")  # persistent if volume mounted at /data
ALLOWED_MASTER_EXTS = {".csv",".xlsx",".xls",".parquet",".pq"}

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
ALIASES_DOMAIN  = ["Domain","Company Domain","Website","Company Website","Company Domain/Website","Domain(2)"]  # optional in working
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
    # de-dupe preserving order
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
    # ensure dtypes
    df_repo = df_repo.copy()
    if "count" in df_repo.columns:
        df_repo["count"] = df_repo["count"].astype("int64")
    df_repo.to_parquet(REPO_PATH, compression="snappy")

def add_examples_to_repo(df_input):
    """
    From any uploaded dataset (master or working), extract rows that HAVE emails,
    parse company/country/domain/pattern, and merge into the persistent repo.
    """
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
    # merge by group and sum counts
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

# =========================
# Build in-memory maps from repo
# =========================
def build_maps_from_repo(df_repo):
    """
    Returns:
      repo_by_domain: {domain -> {pattern -> count}}
      repo_cc: { (company|country) -> {pattern: [domains...] (expanded by count)} }
      repo_c:  { company -> {pattern: [domains...] (expanded by count)} }
    """
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

# =========================
# Learn from current dataset (for extra signal)
# =========================
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

# =========================
# Fill logic
# =========================
def best_from_sources(company_key, country_norm, repo_cc, repo_c, cc_data, c_data):
    # company+country → master repo then data
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

def fill_missing_emails(df, learn_from_this_file=False, output_excel=True):
    df = ensure_cols(df)

    # audit columns
    for col in ["Email_Filled","Email_Fill_Reason","Email_Pattern_Detected","Domain_Guess"]:
        if col not in df.columns: df[col] = pd.NA
    df["Email_Filled"] = df["Email_Filled"].fillna("No")

    # 1) optionally ingest this file's known emails into the persistent repo
    if learn_from_this_file:
        add_examples_to_repo(df)

    # 2) load repo & build maps
    repo_df = load_repo()
    repo_by_domain, repo_cc, repo_c = build_maps_from_repo(repo_df)

    # 3) learn from dataset (extra hints)
    cc_data, c_data = learn_from_dataset(df)

    # 4) fill
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

    # 5) audit existing emails (pattern + domain guess)
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

    # 6) save result
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "emails_filled.xlsx" if output_excel else "emails_filled.csv")
    if output_excel: df.to_excel(out_path, index=False)
    else:            df.to_csv(out_path, index=False)
    return out_path, df

# =========================
# File loaders
# =========================
def load_df(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx",".xlsm",".xltx",".xltm",".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path, keep_default_na=True, low_memory=False)
    if ext in [".parquet",".pq"]:
        return pd.read_parquet(path)
    raise gr.Error(f"Unsupported file type: {ext}")

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Email Format Filler (Persistent Repo)") as demo:
    gr.Markdown("""
    ### Email Format Filler (Persistent Repository)
    - **Tab 1 – Initialize/Replace Master**: Upload your large master once. We parse only the needed columns and persist to disk.
    - **Tab 2 – Fill Missing Emails**: Upload working files anytime. Optionally **learn** from rows that already have emails, then fill the blanks.
    """)

    with gr.Tab("Initialize / Replace Master"):
        master_file = gr.File(label="Master Repository (CSV/XLSX/Parquet)", file_count="single", type="filepath")
        init_btn = gr.Button("Build / Replace Persistent Repository", variant="primary")
        repo_stats = gr.JSON(label="Repository Summary")
        download_repo = gr.File(label="Download Current Repository")

        def init_repo(path):
            if not path: raise gr.Error("Please upload a Master Repository file.")
            ext = os.path.splitext(path)[1].lower()
            if ext not in ALLOWED_MASTER_EXTS:
                raise gr.Error(f"Unsupported master type: {ext}")
            df = load_df(path)
            repo = add_examples_to_repo(df)  # this replaces/merges; if you want REPLACE, delete existing first
            # Save current repo again to ensure persisted
            save_repo(repo)
            # Simple stats
            total_rows = int(repo["count"].sum()) if not repo.empty else 0
            unique_keys = int(repo.shape[0])
            # sample
            tmp = tempfile.mkdtemp()
            repo_path = os.path.join(tmp, "master_repository.parquet")
            repo.to_parquet(repo_path, compression="snappy")
            return {"unique_keys": unique_keys, "total_examples": total_rows}, repo_path

        init_btn.click(init_repo, inputs=[master_file], outputs=[repo_stats, download_repo])

    with gr.Tab("Fill Missing Emails"):
        work_file = gr.File(label="Working Dataset (CSV/XLSX)", file_count="single", type="filepath")
        learn_toggle = gr.Checkbox(value=True, label="Learn from this file (add rows with existing emails to the repository)")
        out_excel = gr.Checkbox(value=True, label="Output Excel (unchecked = CSV)")
        run_btn = gr.Button("Run Fill", variant="primary")
        out_file = gr.File(label="Download Result")
        preview  = gr.Dataframe(label="Preview (first 100 rows)", interactive=False, height=350)

        def run_fill(work_path, learn_from_file, want_excel):
            if not work_path: raise gr.Error("Please upload a Working Dataset.")
            df = load_df(work_path)
            path, result = fill_missing_emails(df, learn_from_this_file=learn_from_file, output_excel=want_excel)
            return path, result.head(100)

        run_btn.click(run_fill, inputs=[work_file, learn_toggle, out_excel], outputs=[out_file, preview])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "8080")))
