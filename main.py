import os
import re
import tempfile
from collections import defaultdict, Counter

import pandas as pd
import gradio as gr
from unidecode import unidecode

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
ALIASES_DOMAIN  = ["Domain","Company Domain","Website","Company Website","Company Domain/Website","Domain(2)"]
ALIASES_COUNTRY = ["Country","Company Country","Office Country","HQ Country","Country/Region"]

SEPARATORS = ["", ".", "_", "-"]
PATTERN_KEYS = [
    "first.last", "f.lastname", "firstname.l", "f.l",
    "firstlast", "flast", "lastfirst", "last.f", "first"
]

# Country → preferred TLD order (first is strongest)
COUNTRY_TLD_PREFS = {
    "ireland": [".ie", ".com"],
    "united kingdom": [".co.uk", ".uk", ".com"],
    "uk": [".co.uk", ".uk", ".com"],
    "england": [".co.uk", ".uk", ".com"],
    "scotland": [".co.uk", ".uk", ".com"],
    "wales": [".co.uk", ".uk", ".com"],
    "germany": [".de", ".com"],
    "france": [".fr", ".com"],
    "spain": [".es", ".com"],
    "italy": [".it", ".com"],
    "netherlands": [".nl", ".com"],
    "belgium": [".be", ".com"],
    "sweden": [".se", ".com"],
    "norway": [".no", ".com"],
    "denmark": [".dk", ".com"],
    "finland": [".fi", ".com"],
    "poland": [".pl", ".com"],
    "portugal": [".pt", ".com"],
    "austria": [".at", ".com"],
    "switzerland": [".ch", ".com"],
    "czech republic": [".cz", ".com"],
    "czechia": [".cz", ".com"],
    "slovakia": [".sk", ".com"],
    "hungary": [".hu", ".com"],
    "romania": [".ro", ".com"],
    "bulgaria": [".bg", ".com"],
    "greece": [".gr", ".com"],
    "turkey": [".com.tr", ".tr", ".com"],
    "united states": [".com", ".us"],
    "usa": [".com", ".us"],
    "canada": [".ca", ".com"],
    "australia": [".com.au", ".au", ".com"],
    "new zealand": [".co.nz", ".nz", ".com"]
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
    f = norm_name(first)
    l = norm_name(last)
    fi = f[:1]
    li = l[:1]

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
    f = norm_name(first)
    l = norm_name(last)
    fi = f[:1]
    li = l[:1]

    def with_sep(coreA, coreB):
        for sep in [".", "_", "-"]:
            return f"{coreA}{sep}{coreB}@{domain}"
        return f"{coreA}{coreB}@{domain}"

    if pattern_key == "first.last": return with_sep(f, l)
    if pattern_key == "f.lastname": return with_sep(fi, l)
    if pattern_key == "firstname.l": return with_sep(f, li)
    if pattern_key == "f.l":        return with_sep(fi, li)
    if pattern_key == "firstlast":  return f"{f}{l}@{domain}"
    if pattern_key == "flast":      return f"{fi}{l}@{domain}"
    if pattern_key == "lastfirst":  return f"{l}{f}@{domain}"
    if pattern_key == "last.f":     return with_sep(l, fi)
    if pattern_key == "first":      return f"{f}@{domain}"
    return None

# =========================
# Ensure needed columns
# =========================
def ensure_cols(df):
    c_company = match_col(ALIASES_COMPANY, df.columns)
    c_email   = match_col(ALIASES_EMAIL, df.columns)
    c_first   = match_col(ALIASES_FIRST, df.columns)
    c_last    = match_col(ALIASES_LAST, df.columns)
    c_domain  = match_col(ALIASES_DOMAIN, df.columns)   # optional
    c_country = match_col(ALIASES_COUNTRY, df.columns)  # optional

    # fallback Name -> (First, Last)
    if c_first is None or c_last is None:
        name_col = match_col(["Name","Full Name"], df.columns)
        if name_col:
            names = df[name_col].fillna("").astype(str)
            first_guess = names.str.split().str[0]
            last_guess  = names.str.split().str[-1]
            if c_first is None:
                df["First Name"] = first_guess
                c_first = "First Name"
            if c_last is None:
                df["Last Name"] = last_guess
                c_last = "Last Name"

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
        **({c_domain: "Domain"} if c_domain else {}),
        **({c_country: "Country"} if c_country else {}),
    })

    df["Company"]    = df["Company"].apply(clean_string)
    df["Email"]      = df["Email"].replace(r"^\s*$", pd.NA, regex=True)
    df["Email"]      = df["Email"].replace(to_replace=["nan","NaN","None","NULL","null"], value=pd.NA)
    df["First Name"] = df["First Name"].apply(clean_string)
    df["Last Name"]  = df["Last Name"].apply(clean_string)

    if "Domain" in df.columns:
        df["Domain"] = df["Domain"].astype(str).str.strip()
        df["Domain"] = df["Domain"].replace(r"^\s*$", pd.NA, regex=True)
        df["Domain"] = df["Domain"].apply(lambda x: re.sub(r"^https?://", "", x) if isinstance(x,str) else x)
        df["Domain"] = df["Domain"].apply(lambda x: x.split("/")[0] if isinstance(x,str) else x)
        df["Domain"] = df["Domain"].str.lower()

    if "Country" in df.columns:
        df["Country"] = df["Country"].apply(normalize_country)

    return df

# =========================
# Repositories (master + data)
# =========================
def choose_best_domain(domains, country_norm):
    if not domains:
        return None
    cnt = Counter(domains)

    if country_norm and country_norm in COUNTRY_TLD_PREFS:
        prefs = COUNTRY_TLD_PREFS[country_norm]
        for pref in prefs:
            candidates = [d for d in cnt if d.endswith(pref)]
            if candidates:
                best = sorted(candidates, key=lambda d: (-cnt[d], d))[0]
                return best

    com_candidates = [d for d in cnt if d.endswith(".com")]
    if com_candidates:
        return sorted(com_candidates, key=lambda d: (-cnt[d], d))[0]

    return sorted(cnt.keys(), key=lambda d: (-cnt[d], d))[0]

def build_pattern_repo_from_master(master_df):
    """
    Returns:
      repo_by_domain: {domain -> {pattern -> count}}
      repo_by_company_country: { (company|country) -> {pattern: [domains]} }
      repo_by_company: { company -> {pattern: [domains]} }
    """
    master_df = ensure_cols(master_df)
    repo_by_domain = defaultdict(lambda: defaultdict(int))
    repo_by_company_country = defaultdict(lambda: defaultdict(list))
    repo_by_company = defaultdict(lambda: defaultdict(list))

    for _, r in master_df.dropna(subset=["Email"]).iterrows():
        first, last, email = r["First Name"], r["Last Name"], str(r["Email"]).strip()
        company = (r["Company"] or "") and str(r["Company"]).strip()
        dom = extract_domain(email)
        pat = detect_email_pattern(first, last, email)
        country = r["Country"] if "Country" in master_df.columns else None
        country_norm = normalize_country(country)

        if pat and dom:
            repo_by_domain[dom][pat] += 1
            if company:
                repo_by_company[company][pat].append(dom)
                if country_norm:
                    key = f"{company}|{country_norm}"
                    repo_by_company_country[key][pat].append(dom)

    return repo_by_domain, repo_by_company_country, repo_by_company

def learn_from_dataset(df):
    pattern_company_country = defaultdict(lambda: defaultdict(list))
    pattern_company = defaultdict(lambda: defaultdict(list))

    for _, r in df.dropna(subset=["Email"]).iterrows():
        first, last, email = r["First Name"], r["Last Name"], str(r["Email"]).strip()
        company = (r["Company"] or "") and str(r["Company"]).strip()
        dom = extract_domain(email)
        pat = detect_email_pattern(first, last, email)
        country_norm = normalize_country(r["Country"]) if "Country" in df.columns else None

        if pat and dom and company:
            pattern_company[company][pat].append(dom)
            if country_norm:
                key = f"{company}|{country_norm}"
                pattern_company_country[key][pat].append(dom)

    return pattern_company_country, pattern_company

# =========================
# Main fill function
# =========================
def fill_missing_emails(df, master_df=None, prefer_domain_repo=True, output_excel=False):
    df = ensure_cols(df)

    # audit columns
    if "Email_Filled" not in df.columns:
        df["Email_Filled"] = "No"
    if "Email_Fill_Reason" not in df.columns:
        df["Email_Fill_Reason"] = ""
    if "Email_Pattern_Detected" not in df.columns:
        df["Email_Pattern_Detected"] = pd.NA
    if "Domain_Guess" not in df.columns:
        df["Domain_Guess"] = pd.NA

    repo_by_domain = defaultdict(lambda: defaultdict(int))
    repo_cc_master = defaultdict(lambda: defaultdict(list))
    repo_c_master  = defaultdict(lambda: defaultdict(list))

    if master_df is not None:
        repo_by_domain, repo_cc_master, repo_c_master = build_pattern_repo_from_master(master_df)

    cc_data, c_data = learn_from_dataset(df)

    def best_from_sources(company, country_norm):
        # try company+country (master then data)
        if country_norm:
            key = f"{company}|{country_norm}"
            for src_tag, src in (("MASTER company+country", repo_cc_master),
                                 ("DATA company+country", cc_data)):
                if key in src and src[key]:
                    fmt_counts = {k: len(v) for k, v in src[key].items() if v}
                    if fmt_counts:
                        pat = max(fmt_counts, key=fmt_counts.get)
                        best_dom = choose_best_domain(src[key][pat], country_norm)
                        if best_dom:
                            return pat, best_dom, src_tag

        # company-only (master then data)
        for src_tag, src in (("MASTER company", repo_c_master),
                             ("DATA company", c_data)):
            if company in src and src[company]:
                fmt_counts = {k: len(v) for k, v in src[company].items() if v}
                if fmt_counts:
                    pat = max(fmt_counts, key=fmt_counts.get)
                    best_dom = choose_best_domain(src[company][pat], country_norm)
                    if best_dom:
                        return pat, best_dom, src_tag

        return None, None, None

    # fill loop
    fill_idxs = df.index[df["Email"].isna()].tolist()
    for idx in fill_idxs:
        row = df.loc[idx]
        company = (row["Company"] or "") and str(row["Company"]).strip()
        first, last = row["First Name"], row["Last Name"]
        country_norm = normalize_country(row["Country"]) if "Country" in df.columns else None
        dom_hint = row["Domain"] if "Domain" in df.columns else None

        pat = None
        domain = None
        src_tag = None

        # 1) exact domain (if given in row and exists in master domain repo)
        if prefer_domain_repo and dom_hint and dom_hint in repo_by_domain:
            dom_pats = repo_by_domain[dom_hint]
            if dom_pats:
                pat = max(dom_pats, key=dom_pats.get)
                domain = dom_hint
                src_tag = "MASTER domain"

        # 2) company+country → company
        if (pat is None or domain is None) and company:
            pat2, dom2, src2 = best_from_sources(company, country_norm)
            if pat2 and dom2:
                pat, domain, src_tag = pat2, dom2, src2

        if pat and domain and first and last:
            new_email = generate_email(first, last, pat, domain)
            if new_email:
                df.at[idx, "Email"] = new_email
                df.at[idx, "Email_Filled"] = "Yes"
                df.at[idx, "Email_Fill_Reason"] = f"Filled via {src_tag} (pattern '{pat}', domain '{domain}')"
                df.at[idx, "Domain_Guess"] = domain

    # audit: pattern detected for rows that already had emails
    detected = []
    for _, r in df.iterrows():
        if pd.isna(r["Email"]):
            detected.append(pd.NA)
        else:
            detected.append(detect_email_pattern(r["First Name"], r["Last Name"], r["Email"]))
            # also backfill Domain_Guess for existing emails
            dom = extract_domain(r["Email"])
            if dom and pd.isna(r.get("Domain_Guess", pd.NA)):
                df.at[_, "Domain_Guess"] = dom
    df["Email_Pattern_Detected"] = detected

    # save
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "emails_filled.xlsx" if output_excel else "emails_filled.csv")
    if output_excel:
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path, df

# =========================
# Gradio UI
# =========================
DESCRIPTION = """
### Email Format Filler (Company + Country aware)
1. Upload your **Working Dataset** (CSV/XLSX) with columns for Company, First Name, Last Name, Email (missing allowed), and *optionally* Country.<br>
2. (Optional) Upload a **Master Repository** (CSV/XLSX/Parquet) with examples (First/Last/Email). The app learns domain & pattern from the email.<br>
3. Click **Run** to fill missing emails. You’ll get a downloadable file plus a preview table.
"""

with gr.Blocks(title="Email Format Filler") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        work_file = gr.File(label="Working Dataset (CSV or Excel)", file_count="single", type="filepath", interactive=True)
        master_file = gr.File(label="Master Repository (optional: CSV/Excel/Parquet)", file_count="single", type="filepath", interactive=True)

    with gr.Row():
        prefer_domain = gr.Checkbox(value=True, label="Prefer Domain-level learning (from Master) when domain present in row")
        want_excel = gr.Checkbox(value=True, label="Output Excel (unchecked = CSV)")

    run_btn = gr.Button("Run", variant="primary")

    out_path = gr.File(label="Download Result")
    preview  = gr.Dataframe(label="Preview (first 100 rows)", interactive=False, wrap=True, height=350)

    def load_df(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xlsx",".xlsm",".xltx",".xltm",".xls"]:
            return pd.read_excel(path)
        if ext == ".csv":
            return pd.read_csv(path, keep_default_na=True, low_memory=False)
        if ext in [".parquet",".pq"]:
            return pd.read_parquet(path)
        raise gr.Error(f"Unsupported file type: {ext}")

    def run_app(work_path, master_path, prefer_domain, want_excel):
        if not work_path:
            raise gr.Error("Please upload a Working Dataset file.")

        df = load_df(work_path)
        master_df = load_df(master_path) if master_path else None

        result_path, result_df = fill_missing_emails(df, master_df, prefer_domain_repo=prefer_domain, output_excel=want_excel)
        return result_path, result_df.head(100)

    run_btn.click(run_app, inputs=[work_file, master_file, prefer_domain, want_excel], outputs=[out_path, preview])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "8080")))
