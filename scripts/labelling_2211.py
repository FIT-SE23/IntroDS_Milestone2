import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import unicodedata

DATA_DIR = Path(r'd:/Coding/KHDL/IntroDS_Milestone2/2211-8001-13746')
OUTPUT_DIR = Path(r'd:/Coding/KHDL/IntroDS_Milestone2/outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_PAPERS = None
THRESHOLD = 1.0


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'[\\{}]', '', text)
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def parse_bibtex(bib_content: str) -> Dict[str, Dict]:
    entries = {}
    entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*?)(?=@|\Z)'
    
    for match in re.finditer(entry_pattern, bib_content, re.DOTALL | re.IGNORECASE):
        entry_type = match.group(1).lower()
        citation_key = match.group(2).strip()
        fields_str = match.group(3)
        
        if entry_type == 'string':
            continue
        
        fields = {}
        field_pattern = r'(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|"([^"]*)"|(\d+))'
        
        for field_match in re.finditer(field_pattern, fields_str, re.IGNORECASE | re.DOTALL):
            field_name = field_match.group(1).lower()
            field_value = next((g for g in field_match.groups()[1:] if g is not None), '')
            fields[field_name] = field_value.strip()
        
        if fields:
            entries[citation_key] = {
                'type': entry_type,
                'title': fields.get('title', ''),
                'author': fields.get('author', ''),
                'year': fields.get('year', ''),
            }
    
    return entries


def load_bib_files(paper_dir: Path) -> Dict[str, Dict]:
    all_entries = {}
    tex_dir = paper_dir / 'tex'
    
    if not tex_dir.exists():
        return all_entries
    
    for bib_file in tex_dir.rglob('*.bib'):
        try:
            if bib_file.stat().st_size > 1000000:
                continue
            with open(bib_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            entries = parse_bibtex(content)
            all_entries.update(entries)
        except Exception as e:
            print(f"Error reading {bib_file}: {e}")
    
    return all_entries


def load_references_json(paper_dir: Path) -> Dict[str, Dict]:
    ref_file = paper_dir / 'references.json'
    if not ref_file.exists():
        return {}
    
    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data is None:
                return {}
            return {k: v for k, v in data.items() if v is not None and isinstance(v, dict)}
    except Exception as e:
        print(f"Error reading {ref_file}: {e}")
        return {}


def calculate_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def match_bibtex_to_arxiv(
    bib_entries: Dict[str, Dict],
    references: Dict[str, Dict],
    threshold: float = 1.0
) -> Tuple[Dict[str, str], List[Dict]]:
    matches = {}
    candidates = []
    
    for cite_key, bib_data in bib_entries.items():
        bib_title = bib_data.get('title', '')
        if not bib_title:
            continue
        
        best_match = None
        best_score = 0.0
        
        for arxiv_id, ref_data in references.items():
            ref_title = ref_data.get('title', '')
            if not ref_title:
                continue
            
            score = calculate_similarity(bib_title, ref_title)
            
            if score > best_score:
                best_score = score
                best_match = arxiv_id
        
        if best_match:
            if best_score >= threshold:
                matches[cite_key] = best_match
            else:
                candidates.append({
                    'citation_key': cite_key,
                    'best_arxiv_id': best_match,
                    'similarity_score': best_score,
                    'bib_title': bib_title[:100],
                    'ref_title': references[best_match].get('title', '')[:100]
                })
    
    return matches, candidates


def main():
    print("="*60)
    print("REFERENCE MATCHING PIPELINE - 2211-80001-13746")
    print("Threshold: 100%")
    print("="*60)
    
    all_paper_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    paper_dirs = all_paper_dirs[:MAX_PAPERS] if MAX_PAPERS else all_paper_dirs
    print(f"Tổng số papers: {len(all_paper_dirs)}")
    print(f"Xử lý: {len(paper_dirs)} papers")
    
    auto_matched = {}
    manual_candidates = {}
    
    papers_with_matches = 0
    total_auto_refs = 0
    
    for idx, paper_dir in enumerate(paper_dirs):
        if idx % 50 == 0:
            print(f"Processing {idx+1}/{len(paper_dirs)}...")
        
        paper_id = paper_dir.name
        bib_entries = load_bib_files(paper_dir)
        references = load_references_json(paper_dir)
        
        if not bib_entries or not references:
            continue
        
        matches, candidates = match_bibtex_to_arxiv(bib_entries, references, threshold=THRESHOLD)
        
        if matches:
            auto_matched[paper_id] = matches
            papers_with_matches += 1
            total_auto_refs += len(matches)
        
        if candidates:
            manual_candidates[paper_id] = candidates
    
    print(f"\n--- KẾT QUẢ ---")
    print(f"Số papers có matches (100%): {papers_with_matches}")
    print(f"Tổng số references tự động match: {total_auto_refs}")
    print(f"Số papers cần label thủ công: {len(manual_candidates)}")
    
    if papers_with_matches >= 5 and total_auto_refs >= 20:
        print("\n✓ ĐÃ ĐẠT YÊU CẦU: >= 5 papers và >= 20 references")
    else:
        print(f"\n⚠ CHƯA ĐẠT YÊU CẦU: Cần 5 papers (có {papers_with_matches}) và 20 refs (có {total_auto_refs})")
    
    output_file_1 = OUTPUT_DIR / 'auto_matched.json'
    with open(output_file_1, 'w', encoding='utf-8') as f:
        json.dump(auto_matched, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {output_file_1}")
    
    output_file_2 = OUTPUT_DIR / 'manual_candidates.json'
    with open(output_file_2, 'w', encoding='utf-8') as f:
        json.dump(manual_candidates, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu: {output_file_2}")
    
    print("\n--- SAMPLE AUTO-MATCHED (5 đầu tiên) ---")
    count = 0
    for paper_id, refs in auto_matched.items():
        if count >= 5:
            break
        print(f"\n{paper_id}:")
        for ref_key, arxiv_id in list(refs.items())[:3]:
            print(f"  {ref_key}: {arxiv_id}")
        if len(refs) > 3:
            print(f"  ... và {len(refs) - 3} refs khác")
        count += 1
    
    print("\n--- THỐNG KÊ CHO MANUAL LABELLING ---")
    total_manual = sum(len(v) for v in manual_candidates.values())
    print(f"Tổng candidates cần xem xét: {total_manual}")
    
    if manual_candidates:
        print("\n--- SAMPLE CANDIDATES ---")
        sample_count = 0
        for paper_id, candidates in manual_candidates.items():
            if sample_count >= 3:
                break
            print(f"\n{paper_id}:")
            for c in candidates[:2]:
                print(f"  [{c['citation_key']}] -> {c['best_arxiv_id']} (score: {c['similarity_score']:.3f})")
                print(f"    BibTeX: {c['bib_title'][:60]}...")
                print(f"    RefJSON: {c['ref_title'][:60]}...")
            sample_count += 1


if __name__ == "__main__":
    main()
