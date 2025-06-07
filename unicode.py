import streamlit as st
import unicodedata
import os
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
from collections import Counter
import datetime
import plotly.express as px
import html

# --- Konfiguracja i pobieranie modeli (wykonywane raz) ---
@st.cache_resource
def load_models():
    """Ładuje modele Hugging Face i cachuje je."""
    try:
        model_name = "gpt2"
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Nie udało się załadować modelu Hugging Face: {e}")
        return None, None

# --- Słowniki i dane do analizy ---
BASIC_HOMOGLYPHS = {
    'A': ['А'], 'a': ['а', 'ɑ'], 'B': ['В'], 'C': ['С'], 'c': ['с'],
    'E': ['Е', 'Ε'], 'e': ['е'], 'H': ['Н'], 'I': ['І', 'Ι'],
    'i': ['і', 'ⅰ'], 'J': ['Ј'], 'j': ['ј'], 'K': ['К'], 'M': ['М'],
    'O': ['О', 'Ο', 'Օ', '0'], 'o': ['о', 'ο'], 'P': ['Р', 'Ρ'], 'p': ['р'],
    'S': ['Ѕ'], 's': ['ѕ'], 'T': ['Т'], 'X': ['Х', 'Χ'], 'x': ['х'],
    'Y': ['Ү', 'Υ'], 'y': ['у'], 'l': ['1', 'І', 'Ӏ'], 'g': ['ɡ'],
}

REVERSE_HOMOGLYPHS = {}
for original, glyph_list in BASIC_HOMOGLYPHS.items():
    for glyph in glyph_list:
        if glyph not in REVERSE_HOMOGLYPHS:
            REVERSE_HOMOGLYPHS[glyph] = []
        REVERSE_HOMOGLYPHS[glyph].append(original)

INVISIBLE_CHAR_PATTERN = re.compile(
    r'[\u00AD\u180E\u200B-\u200F\u202A-\u202E\u2060\u2066-\u2069\uFEFF]'
)
NORMALIZATION_MAP = {
    '‘': "'", '’': "'",
    '“': '"', '”': '"',
    '–': '-', '—': '-',
    '…': '...',
    '\u00A0': ' ', # Non-breaking space
}

# --- Logika Analizy ---

def get_char_script(char):
    try:
        name = unicodedata.name(char).upper()
        if 'LATIN' in name: return 'Latin'
        if 'CYRILLIC' in name: return 'Cyrillic'
        if 'GREEK' in name: return 'Greek'
        if 'ARMENIAN' in name: return 'Armenian'
        if unicodedata.category(char) in ('Zs', 'Po', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Pc'): return 'Common'
    except (ValueError, TypeError): pass
    return 'Unknown'

def analyze_suspicious_patterns(text: str, scientific_mode: bool) -> list:
    anomalies, reported_indices, current_index = [], set(), 0
    words = re.split(r'(\s+)', text)
    for word in words:
        if word.strip():
            scripts_in_word = {get_char_script(c) for c in word if get_char_script(c) != 'Common'}
            if len(scripts_in_word) > 1:
                for i, char in enumerate(word):
                    char_abs_index = current_index + i
                    if char_abs_index not in reported_indices:
                        anomalies.append({"Istotność": "Wysoka", "Problem": "Mieszane alfabety w słowie", "Znak": char, "Indeks": char_abs_index, "Opis": f"Wykryto alfabety: {', '.join(sorted(list(scripts_in_word)))}"})
                        reported_indices.add(char_abs_index)
        current_index += len(word)
    for index, char in enumerate(text):
        if index in reported_indices: continue
        script = get_char_script(char)
        category = unicodedata.category(char)
        
        if char in REVERSE_HOMOGLYPHS:
            anomalies.append({"Istotność": "Wysoka", "Problem": "Wykryto homoglif", "Znak": char, "Indeks": index, "Opis": f"Ten znak może udawać: {', '.join(REVERSE_HOMOGLYPHS[char])}"})
        elif category in {'Cc', 'Cf', 'Co', 'Cs', 'Cn'} or INVISIBLE_CHAR_PATTERN.search(char):
            anomalies.append({"Istotność": "Wysoka", "Problem": "Podejrzana kategoria Unicode", "Znak": f"{char} (U+{ord(char):04X})", "Indeks": index, "Opis": f"Kategoria '{category}' często zawiera znaki niewidoczne."})
        elif script not in ['Latin', 'Common', 'Unknown'] and not (scientific_mode and script == 'Greek') and category not in ('Sm', 'Sc'):
             anomalies.append({"Istotność": "Średnia", "Problem": "Znak spoza alfabetu łacińskiego", "Znak": char, "Indeks": index, "Opis": f"Wykryto znak z alfabetu: {script}"})
        else: continue
        reported_indices.add(index)
    return sorted(anomalies, key=lambda x: x['Indeks'])

def detect_informational_markers(text: str, scientific_mode: bool) -> list:
    markers = []
    for index, char in enumerate(text):
        description, problem = None, None
        category = unicodedata.category(char)
        
        if char in NORMALIZATION_MAP and char != '\u00A0':
            description, problem = f"'{char}' -> '{NORMALIZATION_MAP[char]}'", "Niestandardowa interpunkcja"
        elif scientific_mode and get_char_script(char) == 'Greek':
            try: name = unicodedata.name(char)
            except ValueError: name = "nieznana"
            description, problem = f"Litera grecka ({name})", "Symbol naukowy"
        elif category in ('Sm', 'Sc'):
            try: name = unicodedata.name(char)
            except ValueError: name = "nieznana nazwa"
            description, problem = f"Symbol ({name})", "Symbol specjalny"
        
        if description:
            markers.append({"Istotność": "Informacyjna", "Problem": problem, "Znak": char, "Indeks": index, "Opis": description})
    return sorted(markers, key=lambda x: x['Indeks'])

def detect_stylometric_features(text: str, hf_model, hf_tokenizer) -> dict:
    if not text.strip() or not hf_model: return {"Perpleksja (całość)": "N/A", "Zróżnicowanie (dł. zdań)": "N/A"}
    try:
        inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=hf_model.config.n_positions)
        with torch.no_grad(): outputs = hf_model(inputs.input_ids, labels=inputs.input_ids)
        perplexity = torch.exp(outputs.loss).item()
    except Exception: perplexity = "Błąd obliczeń"
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2: burstiness = 0.0
        else:
            sentence_lengths = [len(hf_tokenizer.tokenize(s)) for s in sentences if s.strip()]
            burstiness = np.std(sentence_lengths) if len(sentence_lengths) >= 2 else 0.0
    except Exception as e:
        st.warning(f"Wewnętrzny błąd podczas obliczania zróżnicowania: {e}"); burstiness = "Błąd obliczeń"
    return {"Perpleksja (całość)": f"{perplexity:.2f}" if isinstance(perplexity, float) else perplexity, "Zróżnicowanie (dł. zdań)": f"{burstiness:.2f}" if isinstance(burstiness, float) else burstiness}

def generate_summary_and_score(anomalies: list) -> dict:
    high_sev_count = sum(1 for a in anomalies if a['Istotność'] == 'Wysoka')
    med_sev_count = sum(1 for a in anomalies if a['Istotność'] == 'Średnia')
    score = high_sev_count * 10 + med_sev_count * 2
    return {"score": score, "high_sev_count": high_sev_count, "med_sev_count": med_sev_count}

def get_final_classification(summary: dict, stylometry: dict, perplexity_threshold: float) -> (str, str):
    score = summary['score']
    try: perplexity = float(stylometry.get("Perpleksja (całość)", 999))
    except (ValueError, TypeError): perplexity = 999
    
    SCORE_HIGH_THRESHOLD, SCORE_MEDIUM_THRESHOLD = 30, 5
    
    if summary['high_sev_count'] > 2: return "Prawdopodobnie wygenerowany przez AI (wykryto silne anomalie)", "red"
    if perplexity < perplexity_threshold and score > SCORE_MEDIUM_THRESHOLD: return "Prawdopodobnie wygenerowany przez AI (niska perpleksja i anomalie)", "red"
    if score >= SCORE_HIGH_THRESHOLD: return "Prawdopodobnie wygenerowany przez AI (wysoki wskaźnik anomalii)", "red"
    if perplexity < perplexity_threshold: return "Prawdopodobnie wygenerowany przez AI (bardzo niska perpleksja)", "orange"
    if score > SCORE_MEDIUM_THRESHOLD: return "Prawdopodobnie zmodyfikowany lub AI (wykryto anomalie)", "orange"
    return "Prawdopodobnie napisany przez człowieka", "green"

def clean_and_normalize_text(text: str) -> (str, dict):
    report = {
        'Niestandardowa interpunkcja': len(re.findall(r'[“”‘’–—…]', text)),
        'Twarde spacje': text.count('\u00A0'),
        'Niewidoczne znaki': len(INVISIBLE_CHAR_PATTERN.findall(text))
    }
    for smart, basic in NORMALIZATION_MAP.items():
        text = text.replace(smart, basic)
    cleaned_text = INVISIBLE_CHAR_PATTERN.sub('', text)
    return cleaned_text, report

# --- Funkcje pomocnicze i UI ---
def read_docx_text(file_stream) -> str:
    try: return "\n".join([para.text for para in docx.Document(file_stream).paragraphs])
    except Exception as e: st.error(f"BŁĄD: Nie udało się odczytać pliku .docx: {e}"); return ""
def read_txt_file(file_stream) -> str:
    try: return file_stream.getvalue().decode("utf-8")
    except Exception as e: st.error(f"BŁĄD: Nie udało się odczytać pliku .txt: {e}"); return ""

def get_context_snippet(text: str, index: int, window: int = 25) -> str:
    char_at_index = text[index]
    start, end = max(0, index - window), min(len(text), index + window)
    prefix = text[start:index].replace('\n', ' ')
    suffix = text[index + 1:end + 1].replace('\n', ' ')

    if not char_at_index.strip() or unicodedata.category(char_at_index) in ('Cc', 'Cf', 'Co'):
        codepoint = f"U+{ord(char_at_index):04X}"
        highlighted_part = f"🚨**`<niewidoczny_znak: {codepoint}>`**🚨"
        return f"...{prefix}{highlighted_part}{suffix}..."
    else:
        highlighted_part = f"**{char_at_index}**"
        return f"...{prefix}{highlighted_part}{suffix}..."

def convert_to_unicode_representation(text: str) -> str:
    return " ".join([f"U+{ord(char):04X}" for char in text])

def display_results_in_table(title: str, results: list, full_text: str):
    df = pd.DataFrame(results)
    if not df.empty and 'Indeks' in df.columns:
        df['Kontekst'] = df.apply(lambda row: get_context_snippet(full_text, row['Indeks']), axis=1)
    def style_severity(row):
        color_map = {'Wysoka': 'background-color: #ffcccb', 'Średnia': 'background-color: #fff3cd', 'Informacyjna': 'background-color: #d1ecf1'}
        return [color_map.get(row['Istotność'], '')] * len(row)
    if not df.empty and 'Istotność' in df.columns:
        st.dataframe(df.style.apply(style_severity, axis=1), use_container_width=True, hide_index=True)
    elif not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

# --- NOWA FUNKCJA DO GENEROWANIA TEKSTU Z ADNOTACJAMI ---
def generate_annotated_text(text: str, anomalies: list, markers: list) -> str:
    """Generuje tekst z adnotacjami w formacie HTML, podświetlając problemy."""
    all_issues = sorted(anomalies + markers, key=lambda x: x['Indeks'])
    issues_by_index = {item['Indeks']: item for item in all_issues}
    
    color_map = {'Wysoka': '#ff4b4b', 'Średnia': '#ffc400', 'Informacyjna': '#2196f3'}
    text_color_map = {'Wysoka': 'white', 'Średnia': 'black', 'Informacyjna': 'white'}

    annotated_html = ""
    i = 0
    text_len = len(text)
    
    while i < text_len:
        if i in issues_by_index:
            issue = issues_by_index[i]
            char_to_process = text[i]
            
            tooltip_title = f"Problem: {issue['Problem']}\nOpis: {issue['Opis']}"
            tooltip_title_escaped = html.escape(tooltip_title, quote=True)

            bg_color = color_map.get(issue['Istotność'], '#f0f2f6')
            text_color = text_color_map.get(issue['Istotność'], 'black')

            display_char = char_to_process
            if not display_char.strip() or unicodedata.category(display_char) in ('Cc', 'Cf', 'Co'):
                 codepoint = f"U+{ord(char_to_process):04X}"
                 display_char = f"<{codepoint[2:]}>"
            
            display_char_escaped = html.escape(display_char)

            annotated_html += f'<span style="background-color: {bg_color}; color: {text_color}; padding: 2px; border-radius: 3px; font-weight: bold; cursor: help;" title="{tooltip_title_escaped}">{display_char_escaped}</span>'
            i += 1
        else:
            try: next_issue_index = min(k for k in issues_by_index if k > i)
            except ValueError: next_issue_index = text_len

            chunk = text[i:next_issue_index]
            annotated_html += html.escape(chunk).replace('\n', '<br>')
            i = next_issue_index
            
    return f'<div style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; border: 1px solid #ccc; padding: 10px; border-radius: 5px; line-height: 1.8;">{annotated_html}</div>'

# --- Główna logika aplikacji Streamlit ---
def main():
    st.set_page_config(page_title="Analizator Tekstu", layout="wide", initial_sidebar_state="expanded")
    model, tokenizer = load_models()

    with st.sidebar:
        st.header("⚙️ Opcje Konfiguracji")
        st.session_state.scientific_mode = st.toggle('Tryb naukowy', help="Dostosowuje analizę do tekstów akademickich, ignorując np. greckie litery jako anomalie.")
        st.session_state.perplexity_threshold = st.slider("Próg perpleksji dla detekcji AI", 30.0, 200.0, 80.0, 5.0, help="Niższe wartości oznaczają, że tylko tekst o bardzo niskiej perpleksji będzie flagowany jako potencjalnie od AI.")
        st.session_state.suspicion_threshold = st.slider("Próg 'Poziomu Podejrzeń'", 10, 100, 50, help="Określa, przy jakiej liczbie punktów ocena zmienia się na 'Wysoką'.")
        with st.expander("💡 Wskazówki dla prac naukowych"): st.markdown("""...""")
        st.text_input("Nazwa raportu", "Raport Analizy Tekstu")
        st.date_input("Data analizy", datetime.date.today())
        if st.button("Wyślij raport (symulacja)"): st.success("Raport został 'wysłany'!")
        st.info("Ta aplikacja jest narzędziem wspomagającym.")

    st.title("🔬 Zaawansowany Analizator Tekstu")
    st.markdown("Wklej tekst lub prześlij plik, aby wykryć anomalie, przeanalizować cechy stylometryczne i otrzymać sugestie poprawek.")

    if 'text_to_analyze' not in st.session_state: st.session_state['text_to_analyze'] = ""
    if 'analysis_done' not in st.session_state: st.session_state['analysis_done'] = False
    if 'modified_text' not in st.session_state: st.session_state['modified_text'] = None

    tab1, tab2 = st.tabs(["📋 Wklej tekst", "📤 Prześlij plik"])
    with tab1:
        text_input = st.text_area("Wprowadź tekst do analizy:", height=250, key="text_area_input", label_visibility="collapsed", placeholder="Wpisz lub wklej tekst tutaj... np. 'Witaj świеciе!' (z cyrylicą 'е')")
        if text_input: st.session_state.text_to_analyze = text_input
    with tab2:
        uploaded_file = st.file_uploader("Wybierz plik `.docx` lub `.txt`", type=["txt", "docx"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.text_to_analyze = read_txt_file(uploaded_file) if uploaded_file.type == "text/plain" else read_docx_text(uploaded_file)
            st.info("Plik został wczytany. Kliknij przycisk 'Analizuj'.")
    
    if st.button("Analizuj Tekst", type="primary", use_container_width=True):
        if not st.session_state.text_to_analyze.strip():
            st.warning("Proszę wkleić tekst lub przesłać plik.", icon="⚠️")
        else:
            with st.spinner('Przeprowadzanie zaawansowanej analizy...'):
                st.session_state.suspicious_patterns = analyze_suspicious_patterns(st.session_state.text_to_analyze, st.session_state.scientific_mode)
                st.session_state.informational_markers = detect_informational_markers(st.session_state.text_to_analyze, st.session_state.scientific_mode)
                st.session_state.stylometric_features = detect_stylometric_features(st.session_state.text_to_analyze, model, tokenizer)
                st.session_state.summary_details = generate_summary_and_score(st.session_state.suspicious_patterns)
                st.session_state.final_verdict, st.session_state.verdict_color = get_final_classification(st.session_state.summary_details, st.session_state.stylometric_features, st.session_state.perplexity_threshold)
            st.session_state.analysis_done = True
            st.session_state.modified_text = None

    if st.session_state.analysis_done:
        st.header("Wyniki Analizy", divider='gray')
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📊 Podsumowanie", "📈 Wizualizacje", "📝 Tekst z Adnotacjami", "🛠️ Narzędzia"])

        with res_tab1:
            st.markdown(f"### Ostateczna ocena: :{st.session_state.verdict_color}[{st.session_state.final_verdict}]")
            score = st.session_state.summary_details["score"]
            if score == 0: assessment, color = "Bardzo Niski", "green"
            elif score < 20: assessment, color = "Niski", "blue"
            elif score < st.session_state.suspicion_threshold: assessment, color = "Średni", "orange"
            else: assessment, color = "Wysoki", "red"
            st.subheader(f"Poziom Podejrzeń (na podst. anomalii): :{color}[{assessment}]")
            st.progress(min(score, 100))
            c1, c2 = st.columns(2)
            c1.metric("Anomalie o wysokiej istotności", st.session_state.summary_details["high_sev_count"], help="Problemy takie jak mieszane alfabety lub homoglify.")
            c2.metric("Anomalie o średniej istotności", st.session_state.summary_details["med_sev_count"], help="Problemy takie jak znaki spoza alfabetu łacińskiego.")
            st.subheader("Zaawansowane Metryki Stylometryczne")
            if model:
                cols_metrics = st.columns(len(st.session_state.stylometric_features))
                for i, (metric, value) in enumerate(st.session_state.stylometric_features.items()):
                    cols_metrics[i].metric(metric, value)
            else: st.error("Model językowy nie został załadowany.")
        
        with res_tab2:
            st.subheader("Wizualizacja znalezionych problemów", divider='rainbow')
            all_issues = st.session_state.suspicious_patterns + st.session_state.informational_markers
            if all_issues:
                issues_df = pd.DataFrame(all_issues)
                st.markdown("**Mapa Cieplna Anomalii w Tekście**")
                text_len = len(st.session_state.text_to_analyze)
                if text_len > 0:
                    bins = np.linspace(0, text_len, 11); labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(10)]
                    issues_df['Segment'] = pd.cut(issues_df['Indeks'], bins=bins, labels=labels, right=False, include_lowest=True)
                    heatmap_data = issues_df['Segment'].value_counts().sort_index()
                    st.bar_chart(heatmap_data, use_container_width=True)
                else: st.info("Tekst jest zbyt krótki, aby wygenerować mapę cieplną.")
                st.markdown("**Podział Typów Problemów**")
                problem_counts = issues_df['Problem'].value_counts()
                fig_donut = px.pie(problem_counts, values=problem_counts.values, names=problem_counts.index, hole=.4, title="Procentowy udział wszystkich znalezionych elementów")
                st.plotly_chart(fig_donut, use_container_width=True)
            else: st.success("Nie znaleziono żadnych anomalii ani znaczników do wizualizacji.")

        with res_tab3:
            st.subheader("Interaktywna analiza tekstu")
            st.info("Najechanie na podświetlony element pokaże szczegóły wykrytego problemu. Pomaga to zrozumieć, gdzie dokładnie w tekście występują anomalie.")
            annotated_html = generate_annotated_text(st.session_state.text_to_analyze, st.session_state.suspicious_patterns, st.session_state.informational_markers)
            st.markdown(annotated_html, unsafe_allow_html=True)

        with res_tab4:
            st.subheader("Sugestie i narzędzia do poprawy tekstu")
            st.markdown("**Oczyszczanie i Normalizacja Tekstu**")
            cleaned_text, report = clean_and_normalize_text(st.session_state.text_to_analyze)
            total_issues_to_clean = sum(report.values())
            if total_issues_to_clean == 0:
                st.info("Nie wykryto znaków do automatycznego oczyszczenia.")
            else:
                st.warning(f"Wykryto {total_issues_to_clean} znaków, które można automatycznie oczyścić i znormalizować.")
                with st.expander("Zobacz szczegóły znaków do oczyszczenia"):
                    for key, value in report.items():
                        if value > 0:
                            if key == 'Niestandardowa interpunkcja': st.markdown(f"- **{key} ({value}):** Znaki takie jak `“”‘’` czy pauzy `–—` zostaną zamienione na standardowe odpowiedniki ASCII.")
                            elif key == 'Twarde spacje': st.markdown(f"- **{key} ({value}):** Znaki `U+00A0` (twarde spacje) zostaną zamienione na standardowe spacje.")
                            elif key == 'Niewidoczne znaki': st.markdown(f"- **{key} ({value}):** Znaki takie jak spacja o zerowej szerokości (`U+200B`) zostaną usunięte.")
                if st.button("Wyczyść i Znormalizuj Tekst"):
                    st.session_state.modified_text = cleaned_text
                    st.success("Tekst został oczyszczony!"); st.rerun()
            
            st.markdown("---")
            st.markdown("**Interaktywny edytor homoglifów**")
            replaceable = [a for a in st.session_state.suspicious_patterns if a['Problem'] in ["Wykryto homoglif", "Mieszane alfabety w słowie"]]
            if not replaceable: st.info("Nie znaleziono homoglifów do podmiany.")
            else:
                with st.form("replace_form"):
                    for item in replaceable:
                        replacement = REVERSE_HOMOGLYPHS.get(item['Znak'], ['?'])[0]
                        label = f"Podmień '{item['Znak']}' na '{replacement}' w kontekście: {get_context_snippet(st.session_state.text_to_analyze, item['Indeks'])}"
                        st.checkbox(label, key=f"replace_{item['Indeks']}")
                    if st.form_submit_button("Zastosuj wybrane zmiany w homoglifach"):
                        text_to_modify = st.session_state.modified_text or st.session_state.text_to_analyze
                        text_list = list(text_to_modify)
                        modified_count = 0
                        for item in replaceable:
                            if st.session_state.get(f"replace_{item['Indeks']}", False):
                                replacement = REVERSE_HOMOGLYPHS.get(item['Znak'], ['?'])[0]
                                text_list[item['Indeks']] = replacement; modified_count += 1
                        st.session_state.modified_text = "".join(text_list)
                        st.success(f"Zastosowano zmiany! Podmieniono {modified_count} homoglifów.")
            
            if st.session_state.modified_text:
                st.subheader("Tekst po modyfikacji")
                st.text_area("Oczyszczony tekst:", st.session_state.modified_text, height=200, disabled=True)
                st.download_button("Pobierz oczyszczony tekst", st.session_state.modified_text.encode('utf-8'), "oczyszczony_tekst.txt", "text/plain")

if __name__ == "__main__":
    main()
