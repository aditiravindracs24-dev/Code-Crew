
"""
SpecCraft - Generic TF-IDF SRS Extractor (No external models)
Revisions:
 - Automatic actor detection from uploaded files (keyword-based)
 - No hard-coded actors; show "None detected" when none found
 - Plain-text PDF export using ReportLab
 - Multi-file upload supported
"""

import streamlit as st
st.markdown("""
    <style>
        .neon-header-box {
            background: linear-gradient(135deg, #ff0099, #ff66cc);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 0 25px #ff33aa, 0 0 45px #ff0088;
            margin-bottom: 25px;
            border: 2px solid #ffb3e6;
        }

        .neon-header-box h1 {
            color: #ffffff;
            font-weight: 900;
            font-size: 45px;
            text-shadow: 0 0 12px #ffffff, 0 0 20px #ffb3f0, 0 0 35px #ff00aa;
            margin-bottom: 12px;
        }

        .neon-header-box p,
        .neon-header-box li {
            color: #fff0fb;
            font-size: 20px;
            font-weight: 700;  /* BOLD */
            text-shadow: 0 0 6px #ffccee;
        }

        .neon-header-box ul {
            margin-left: 20px;
        }
    </style>

    <div class="neon-header-box">
        <h1>✨ SpecCraft – Enhanced SRS Extractor</h1>
        <p><b>Upload a PDF / DOCX / TXT / JPG / PNG and the app will:</b></p>
        <ul>
            <li><b>Extract text (OCR for images / scanned PDFs)</b></li>
            <li><b>Split text into sentences/clauses</b></li>
            <li><b>Classify into Functional / Non-Functional / Constraints</b></li>
            <li><b>Deduplicate and rewrite into clean SRS phrasing</b></li>
            <li><b>Group requirements by category</b></li>
            <li><b>Export Markdown and plain-text PDF</b></li>
        </ul>
    </div>
""", unsafe_allow_html=True)
import fitz  
import docx
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import pytesseract
import re
import math
import os
import tempfile
from typing import List, Tuple, Dict


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4




ACTOR_KEYWORDS = [
    "user", "customer", "client", "admin", "student", "teacher", "doctor",
    "patient", "researcher", "employee", "manager", "operator", "player",
    "buyer", "seller", "driver", "passenger", "applicant", "resident", "vendor"
]

def extract_actors(text: str) -> List[str]:
    """
    Keyword-based actor detection.
    Returns a sorted list of unique actors (capitalized) found in the text.
    If none found, returns an empty list.
    """
    actors = set()
    if not text:
        return []

    lower = text.lower()

    
    for a in ACTOR_KEYWORDS:
        if re.search(r'\b' + re.escape(a) + r's?\b', lower):
            actors.add(a.capitalize())

 
    title_matches = re.findall(r'\b([A-Z][a-zA-Z]{1,30})\b', text)
    for word in title_matches:
        if word.lower() in ACTOR_KEYWORDS:
            actors.add(word)

    return sorted(list(actors))



def is_image_file(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"))

def ocr_image_file(file) -> str:
    file.seek(0)
    try:
        img = Image.open(file)
    except Exception:
        file.seek(0)
        img = Image.open(file)
    img = img.convert("RGB")
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)
    text = pytesseract.image_to_string(img, config="--psm 6")
    if len(text.strip()) < 8:
        text = pytesseract.image_to_string(img, config="--psm 4")
    return text

def extract_text_from_file(uploaded_file) -> str:
    name = getattr(uploaded_file, "name", "uploaded")
    lower = name.lower()
    try:
        if lower.endswith(".pdf"):
            uploaded_file.seek(0)
            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pages = []
            for p in pdf:
                t = p.get_text()
                if not t or len(t.strip()) < 20:
                    pix = p.get_pixmap(dpi=150)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    t = pytesseract.image_to_string(img, config="--psm 6")
                pages.append(t or "")
            return "\n".join(pages)

        if lower.endswith(".docx"):
            uploaded_file.seek(0)
            data = uploaded_file.read()
            doc = docx.Document(BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)

        if lower.endswith(".txt"):
            uploaded_file.seek(0)
            return uploaded_file.read().decode("utf-8", errors="ignore")

        if is_image_file(lower):
            uploaded_file.seek(0)
            return ocr_image_file(uploaded_file)

        
        uploaded_file.seek(0)
        try:
            return ocr_image_file(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            try:
                return raw.decode("utf-8", errors="ignore")
            except Exception:
                return ""

    except Exception as e:
        return f"[Error extracting {name}: {e}]"



SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
CLAUSE_SPLIT_WORDS = [' and ', ' or ', ';', ' which ', ' that ', ' while ', ', however', ', and', ':']

def split_to_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    parts = SPLIT_RE.split(text)
    clean = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 280:
            subs = split_into_clauses(p)
            clean.extend([s.strip() for s in subs if len(s.strip()) > 4])
        else:
            clean.append(p)
    return [s for s in clean if len(s) > 6]

def split_into_clauses(sentence: str) -> List[str]:
    s = sentence
    clauses = [s]
    for token in CLAUSE_SPLIT_WORDS:
        new = []
        for c in clauses:
            if token in c.lower():
                parts = [p.strip() for p in re.split(re.escape(token), c) if p.strip()]
                new.extend(parts)
            else:
                new.append(c)
        clauses = new
    out = []
    for c in clauses:
        if len(c) < 20 and out:
            out[-1] += " " + c
        else:
            out.append(c)
    return out



def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks

def build_vocab_and_df(docs: List[str], min_df: int = 1):
    df = {}
    total = 0
    for doc in docs:
        total += 1
        toks = set(tokenize(doc))
        for t in toks:
            df[t] = df.get(t, 0) + 1
    vocab = {}
    idx = 0
    for term, cnt in sorted(df.items(), key=lambda x: (-x[1], x[0])):
        if cnt >= min_df:
            vocab[term] = idx
            idx += 1
    return vocab, df, total

def compute_tfidf_rows(docs: List[str], vocab: Dict[str,int], df: Dict[str,int], total_docs: int):
    idf = {}
    for term, idx in vocab.items():
        idf[term] = math.log((1 + total_docs) / (1 + df.get(term, 0))) + 1.0
    rows = []
    for doc in docs:
        toks = tokenize(doc)
        tf = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            rows.append([0.0]*len(vocab))
            continue
        max_tf = max(tf.values())
        vec = [0.0]*len(vocab)
        for term, cnt in tf.items():
            tf_scaled = 0.5 + 0.5 * (cnt / max_tf)
            vec[vocab[term]] = tf_scaled * idf[term]
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        rows.append(vec)
    return rows

def cosine_vec(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    return sum(x*y for x,y in zip(a,b))



FR_TEMPLATES = [
    "the system shall", "the system must", "allow the user to", "user can",
    "provide", "perform", "execute", "support", "process", "submit", "register"
]
NFR_TEMPLATES = [
    "performance", "security", "availability", "usability", "reliability",
    "scalability", "efficiency", "latency", "throughput", "privacy", "battery life"
]
CONS_TEMPLATES = [
    "constraint", "limitation", "restriction", "must not", "cannot", "only",
    "depends on", "bound by", "limited to", "disqualify", "disqualification"
]



def classify_sentences(sentences: List[str], top_vocab_size:int=2000,
                       dedupe_threshold: float=0.72, classify_threshold: float=0.16):
   
    if not sentences:
        return {"Actor": [], "Functional Requirement": [], "Non-Functional Requirement": [], "Constraint": []}

    vocab, df, total = build_vocab_and_df(sentences, min_df=1)
    sent_vecs = compute_tfidf_rows(sentences, vocab, df, total)

    kept_sents, kept_vecs = dedupe_pass(sentences, sent_vecs, threshold=dedupe_threshold)

    templates = FR_TEMPLATES + NFR_TEMPLATES + CONS_TEMPLATES
    combined_docs = kept_sents + templates
    vocab2, df2, total2 = build_vocab_and_df(combined_docs, min_df=1)
    kept_vecs2 = compute_tfidf_rows(kept_sents, vocab2, df2, total2)
    template_vecs = compute_tfidf_rows(templates, vocab2, df2, total2)
    labels = (["fr"] * len(FR_TEMPLATES)) + (["nfr"] * len(NFR_TEMPLATES)) + (["cons"] * len(CONS_TEMPLATES))

    fr_list, nfr_list, cons_list = [], [], []

    for i, sent in enumerate(kept_sents):
        vec = kept_vecs2[i]
        sims = [cosine_vec(vec, tvec) for tvec in template_vecs] if template_vecs else [0.0]
        best_idx = max(range(len(sims)), key=lambda j: sims[j]) if sims else 0
        best_sim = sims[best_idx] if sims else 0.0
        candidate_label = labels[best_idx] if sims else None

        kscore = keyword_score(sent)

        score_fr = (best_sim if candidate_label=="fr" else 0.0) + 0.6*kscore.get("fr",0)
        score_nfr = (best_sim if candidate_label=="nfr" else 0.0) + 0.6*kscore.get("nfr",0)
        score_cons = (best_sim if candidate_label=="cons" else 0.0) + 0.6*kscore.get("cons",0)

        best_final = max(("fr", score_fr), ("nfr", score_nfr), ("cons", score_cons), key=lambda x: x[1])

        if best_final[1] < classify_threshold:
            lab = quick_regex_label(sent)
        else:
            lab = best_final[0]

        if lab == "fr":
            fr_list.append(sent)
        elif lab == "nfr":
            nfr_list.append(sent)
        elif lab == "cons":
            cons_list.append(sent)

    fr_final = normalized_dedupe(fr_list)
    nfr_final = normalized_dedupe(nfr_list)
    cons_final = normalized_dedupe(cons_list)

    fr_final = merge_similar_texts(fr_final)
    nfr_final = merge_similar_texts(nfr_final)
    cons_final = merge_similar_texts(cons_final)

    return {
        "Actor": [],  
        "Functional Requirement": fr_final,
        "Non-Functional Requirement": nfr_final,
        "Constraint": cons_final
    }



FR_KEYWORDS = ["must", "should", "shall", "allow", "enable", "provide", "perform", "execute", "support", "process", "submit", "register"]
NFR_KEYWORDS = ["performance", "security", "availability", "usability", "reliability", "scalability", "efficiency", "latency", "privacy", "battery"]
CONS_KEYWORDS = ["limited", "only", "restricted", "cannot", "must not", "depends", "constraint", "disqualify", "disqualification"]

def keyword_score(sent: str) -> Dict[str, float]:
    s = sent.lower()
    score = {"fr":0.0, "nfr":0.0, "cons":0.0}
    for k in FR_KEYWORDS:
        if re.search(r'\b' + re.escape(k) + r'\b', s):
            score["fr"] += 1.0
    for k in NFR_KEYWORDS:
        if re.search(r'\b' + re.escape(k) + r'\b', s):
            score["nfr"] += 1.0
    for k in CONS_KEYWORDS:
        if re.search(r'\b' + re.escape(k) + r'\b', s):
            score["cons"] += 1.0
    maxv = max(score.values()) if score else 1.0
    if maxv > 0:
        score = {k:v/maxv for k,v in score.items()}
    return score

def quick_regex_label(sent: str) -> str:
    s = sent.lower()
    if re.search(r'\b(limited|only|restricted|cannot|must not|depends|disqual)\b', s):
        return "cons"
    if re.search(r'\b(performance|security|usability|reliability|scalability|availability|privacy|battery)\b', s):
        return "nfr"
    if re.search(r'\b(must|should|shall|allow|enable|user can|provide|support|process|submit|register)\b', s):
        return "fr"
    return "fr"



def dedupe_pass(sentences: List[str], vectors: List[List[float]], threshold: float=0.72) -> Tuple[List[str], List[List[float]]]:
    kept = []
    kept_vecs = []
    for i, s in enumerate(sentences):
        vec = vectors[i]
        dup = False
        if kept_vecs:
            sims = [cosine_vec(vec, kv) for kv in kept_vecs]
            if max(sims) >= threshold:
                dup = True
        if not dup:
            kept.append(s)
            kept_vecs.append(vec)
    return kept, kept_vecs

def normalized_dedupe(lst: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in lst:
        key = re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out

def merge_similar_texts(lst: List[str], overlap_threshold: float=0.65) -> List[str]:
    out = []
    used = [False]*len(lst)
    toks = [set(tokenize(s)) for s in lst]
    for i, ti in enumerate(toks):
        if used[i]: continue
        group = [i]
        for j in range(i+1, len(toks)):
            if used[j]: continue
            inter = ti.intersection(toks[j])
            denom = min(len(ti), len(toks[j])) if min(len(ti), len(toks[j]))>0 else 1
            if len(inter)/denom >= overlap_threshold:
                group.append(j)
                used[j] = True
        reps = [lst[k] for k in group]
        rep = max(reps, key=lambda x: len(x))
        out.append(rep)
    return out



def rewrite_fr(sent: str) -> str:
    s = sent.strip().rstrip('.')
    s = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', s)
    verbs = ["provide", "allow", "enable", "support", "perform", "execute", "process", "submit", "register", "detect", "predict"]
    for v in verbs:
        if re.search(r'\b' + re.escape(v) + r'\b', s.lower()):
            return f"The system shall {s}."
    return f"The system shall {s} to enable effective operation and to meet the user’s needs."

def rewrite_nfr(sent: str) -> str:
    s = sent.strip().rstrip('.')
    s = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', s)
    if re.search(r'\bsecurity\b', s.lower()):
        return f"The system shall ensure data security and confidentiality by employing appropriate encryption, access controls, and auditing mechanisms."
    if re.search(r'\bprivacy\b', s.lower()):
        return f"The system shall ensure user privacy by minimizing data collection, supporting user consent, and protecting personal data in transit and at rest."
    if re.search(r'\bbattery\b|\bpower\b', s.lower()):
        return f"The system shall ensure energy efficiency and provide sufficient battery life to support continuous operation under expected usage."
    if re.search(r'\busability\b|\bcomfort\b', s.lower()):
        return f"The system shall ensure high usability and comfort for end users, with intuitive interfaces and ergonomic design."
    return f"The system shall ensure {s} to meet quality and reliability expectations."

def rewrite_constraint(sent: str) -> str:
    s = sent.strip().rstrip('.')
    s = re.sub(r'^[\-\u2022\*\d\.\)\s]+', '', s)
    if re.search(r'\b(cannot|must not|not allowed|no)\b', s.lower()):
        core = re.sub(r'\b(cannot|must not|not allowed|no)\b', '', s, flags=re.I).strip()
        return f"The system must not {core}."
    if re.search(r'\b(private|restricted|limited|only)\b', s.lower()):
        return f"The system must operate within the specified restrictions and comply with the stated limitations: {s}."
    return f"The system must {s}."



GROUPS = {
    "Hardware Requirements": ["battery", "sensor", "patch", "hardware", "power", "flexible", "form factor", "nanogenerator"],
    "Software & AI Requirements": ["ai", "algorithm", "model", "app", "mobile", "analytics", "predict", "learning", "dashboard", "ar "],
    "Data & Privacy Requirements": ["data", "privacy", "security", "encrypt", "consent", "share", "transmit"],
    "Usability Requirements": ["usability", "comfort", "ui", "ux", "responsive", "interface", "easy", "intuitive"],
    "Clinical & Regulatory Requirements": ["clinical", "regulatory", "accuracy", "standard", "certify", "approved", "medical"],
    "Constraints": ["must", "must not", "limited", "only", "restricted", "cannot", "depends"]
}

def assign_group(sent: str) -> str:
    s = sent.lower()
    scores = {g:0 for g in GROUPS}
    for g, kws in GROUPS.items():
        for k in kws:
            if k in s:
                scores[g] += 1
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        if re.search(r'\b(sensor|patch|battery|hardware)\b', s):
            return "Hardware Requirements"
        if re.search(r'\b(security|data|privacy|encrypt)\b', s):
            return "Data & Privacy Requirements"
        if re.search(r'\b(clinical|accuracy|regulatory)\b', s):
            return "Clinical & Regulatory Requirements"
        return "Software & AI Requirements"
    return best[0]



def generate_pdf_from_md(md_text: str) -> str:
    """
    Generate a plain-text PDF from markdown-like text using ReportLab.
    Returns path to temporary PDF file.
    """
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
    story = []

    for line in md_text.split("\n"):
        clean_line = line.replace("**", "").replace("#", "")
       
        if not clean_line.strip():
            story.append(Paragraph(" ", style))
        else:
            
            story.append(Paragraph(clean_line, style))

    doc.build(story)
    return temp_pdf.name



def process_text_to_srs(raw_text: str):
   
    actors = extract_actors(raw_text)


    sentences = split_to_sentences(raw_text)

    
    classified = classify_sentences(sentences)
    frs = classified.get("Functional Requirement", [])
    nfrs = classified.get("Non-Functional Requirement", [])
    cons = classified.get("Constraint", [])

   
    rewritten_frs = [rewrite_fr(s) for s in frs]
    rewritten_nfrs = [rewrite_nfr(s) for s in nfrs]
    rewritten_cons = [rewrite_constraint(s) for s in cons]

  
    groups = {
        "Hardware Requirements": [],
        "Software & AI Requirements": [],
        "Data & Privacy Requirements": [],
        "Usability Requirements": [],
        "Clinical & Regulatory Requirements": [],
        "Constraints": []
    }

    for r in rewritten_frs:
        g = assign_group(r)
        groups[g].append(r)
    for r in rewritten_nfrs:
        g = assign_group(r)
        groups[g].append(r)
    for r in rewritten_cons:
        groups["Constraints"].append(r)

    
    md = "# SOFTWARE REQUIREMENTS SPECIFICATION (SRS)\n\n"

    
    md += "## Actors\n"
    if actors:
        for a in actors:
            md += f"- {a}\n"
    else:
        md += "- None detected\n"
    md += "\n"

    md += "## Functional Requirements (Rewritten)\n"
    count = 1
    for gname in ["Hardware Requirements", "Software & AI Requirements", "Data & Privacy Requirements", "Usability Requirements", "Clinical & Regulatory Requirements"]:
        items = groups[gname]
        if items:
            md += f"\n### {gname}\n"
            for it in items:
                md += f"- **FR-{count:03d}**: {it}\n"
                count += 1

    md += "\n## Non-Functional Requirements (Rewritten)\n"
    ncount = 1
    for gname in ["Hardware Requirements", "Software & AI Requirements", "Data & Privacy Requirements", "Usability Requirements", "Clinical & Regulatory Requirements"]:
        items = groups[gname]
        nfr_items = [it for it in items if it.lower().startswith("the system shall ensure") or any(k in it.lower() for k in NFR_KEYWORDS)]
        if nfr_items:
            md += f"\n### {gname}\n"
            for it in nfr_items:
                md += f"- **NFR-{ncount:03d}**: {it}\n"
                ncount += 1

    leftover_nfrs = [it for it in rewritten_nfrs if it not in sum(groups.values(), [])]
    if leftover_nfrs:
        md += "\n### Other NFRs\n"
        for it in leftover_nfrs:
            md += f"- **NFR-{ncount:03d}**: {it}\n"
            ncount += 1

    md += "\n## Constraints (Rewritten)\n"
    ccount = 1
    for c in groups["Constraints"]:
        md += f"- **C-{ccount:03d}**: {c}\n"
        ccount += 1

    leftover_cons = [it for it in rewritten_cons if it not in groups["Constraints"]]
    for c in leftover_cons:
        md += f"- **C-{ccount:03d}**: {c}\n"
        ccount += 1

    md += "\n## Appendix - Raw Extracted Text (truncated)\n"
    md += raw_text[:3000] + ("..." if len(raw_text) > 3000 else "")

    return md



st.markdown("### Upload files (PDF / DOCX / TXT / PNG / JPG).")
uploaded_files = st.file_uploader("Upload files", type=["pdf","docx","txt","png","jpg","jpeg"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Extracting text from uploaded files..."):
        combined_text = ""
        errors = []
        for f in uploaded_files:
            try:
                t = extract_text_from_file(f)
                combined_text += t + "\n\n"
            except Exception as e:
                errors.append((getattr(f, "name", "unknown"), str(e)))
        if errors:
            st.warning("Some files failed to extract. See details below.")
            for n, e in errors:
                st.write(f"- {n}: {e}")

    if not combined_text.strip():
        st.error("No text could be extracted from uploaded files.")
    else:
        with st.spinner("Processing text into SRS..."):
            srs_md = process_text_to_srs(combined_text)
        st.success("SRS generated.")
        st.markdown("### 📄 Generated SRS (rewritten only)")
        st.markdown(f"<div style='background:#0b0b0b;padding:12px;border-radius:8px'><pre style='white-space:pre-wrap;color:#eee'>{srs_md}</pre></div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download SRS (Markdown)", srs_md, file_name="generated_srs.md")

      
        try:
            pdf_path = generate_pdf_from_md(srs_md)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="⬇️ Download SRS (PDF)",
                    data=pdf_file,
                    file_name="generated_srs.pdf",
                    mime="application/pdf"
                )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

     
        st.markdown("### 🔎 Summary")
        fr_count = srs_md.count("**FR-")
        nfr_count = srs_md.count("**NFR-")
        cons_count = srs_md.count("**C-")
        st.write(f"- Functional Requirements: {fr_count}")
        st.write(f"- Non-Functional Requirements: {nfr_count}")
        st.write(f"- Constraints: {cons_count}")

else:
    st.info("Upload one or more files to extract and generate the SRS.")
