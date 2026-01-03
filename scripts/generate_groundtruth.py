"""
Reference Matching Pipeline - Ground Truth Generation Script

Script này thực hiện:
1. Parse file BibTeX từ thư mục tex/ của mỗi paper
2. Đọc references.json để lấy arXiv IDs và metadata
3. Matching các BibTeX entries với arXiv ID dựa trên title/author similarity
4. Xuất ra file pred.json theo format yêu cầu

Usage:
    python generate_groundtruth.py --data_dir <path_to_data> --output <output_path> --num_papers <n>
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import unicodedata

def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, remove diacritics, remove punctuation."""
    if not text:
        return ""
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'[\\{}]', '', text)
    # Lowercase
    text = text.lower()
    # Remove accents/diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def parse_bibtex(bib_content: str) -> Dict[str, Dict]:
    """
    Parse BibTeX content và trả về dictionary của các entries.
    
    Returns:
        Dict[str, Dict]: {citation_key: {title: ..., author: ..., year: ...}}
    """
    entries = {}
    
    # Pattern để match BibTeX entries
    entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^@]*?)(?=@|\Z)'
    
    for match in re.finditer(entry_pattern, bib_content, re.DOTALL | re.IGNORECASE):
        entry_type = match.group(1).lower()
        citation_key = match.group(2).strip()
        fields_str = match.group(3)
        
        # Skip STRING entries
        if entry_type == 'string':
            continue
        
        # Parse fields
        fields = {}
        # Pattern để match field = {...} hoặc field = "..."
        field_pattern = r'(\w+)\s*=\s*(?:\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|"([^"]*)"|\{([^}]*)\}|(\d+))'
        
        for field_match in re.finditer(field_pattern, fields_str, re.IGNORECASE | re.DOTALL):
            field_name = field_match.group(1).lower()
            # Get the first non-None captured group for value
            field_value = next((g for g in field_match.groups()[1:] if g is not None), '')
            field_value = field_value.strip()
            fields[field_name] = field_value
        
        if fields:
            entries[citation_key] = {
                'type': entry_type,
                'title': fields.get('title', ''),
                'author': fields.get('author', ''),
                'year': fields.get('year', ''),
                'booktitle': fields.get('booktitle', ''),
                'journal': fields.get('journal', '')
            }
    
    return entries

def load_bib_files(paper_dir: Path) -> Dict[str, Dict]:
    """
    Load tất cả file .bib trong thư mục tex/ của một paper.
    
    Returns:
        Dict[str, Dict]: Combined dictionary của tất cả BibTeX entries
    """
    all_entries = {}
    tex_dir = paper_dir / 'tex'
    
    if not tex_dir.exists():
        return all_entries
    
    # Tìm tất cả file .bib recursively
    for bib_file in tex_dir.rglob('*.bib'):
        try:
            # Skip files quá lớn (có thể là consolidated bibliography)
            if bib_file.stat().st_size > 1000000:  # Skip files > 1MB
                continue
                
            with open(bib_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            entries = parse_bibtex(content)
            all_entries.update(entries)
        except Exception as e:
            print(f"Error reading {bib_file}: {e}")
    
    return all_entries

def load_references_json(paper_dir: Path) -> Dict[str, Dict]:
    """Load references.json và trả về dictionary với arXiv ID là key."""
    ref_file = paper_dir / 'references.json'
    if not ref_file.exists():
        return {}
    
    try:
        with open(ref_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure we return a dict and filter out None values
            if data is None:
                return {}
            # Filter out entries where value is None or not a dict
            return {k: v for k, v in data.items() if v is not None and isinstance(v, dict)}
    except Exception as e:
        print(f"Error reading {ref_file}: {e}")
        return {}

def calculate_similarity(text1: str, text2: str) -> float:
    """Tính similarity score giữa hai chuỗi sử dụng SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def match_bibtex_to_arxiv(
    bib_entries: Dict[str, Dict],
    references: Dict[str, Dict],
    threshold: float = 0.7
) -> Dict[str, str]:
    """
    Match BibTeX entries với arXiv IDs dựa trên title similarity.
    
    Args:
        bib_entries: Dictionary của BibTeX entries
        references: Dictionary từ references.json
        threshold: Ngưỡng similarity tối thiểu để xem là match
    
    Returns:
        Dict[str, str]: {citation_key: arxiv_id} cho các matches đạt ngưỡng
    """
    matches = {}
    
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
            
            # Calculate title similarity
            score = calculate_similarity(bib_title, ref_title)
            
            if score > best_score:
                best_score = score
                best_match = arxiv_id
        
        if best_match and best_score >= threshold:
            matches[cite_key] = best_match
    
    return matches

def process_paper(paper_dir: Path, threshold: float = 0.7) -> Tuple[Dict[str, str], Dict]:
    """
    Xử lý một paper và trả về groundtruth matches cùng với statistics.
    
    Returns:
        Tuple[Dict[str, str], Dict]: (matches, stats)
    """
    # Load data
    bib_entries = load_bib_files(paper_dir)
    references = load_references_json(paper_dir)
    
    stats = {
        'paper_id': paper_dir.name,
        'num_bib_entries': len(bib_entries),
        'num_references': len(references),
        'num_matches': 0
    }
    
    if not bib_entries or not references:
        return {}, stats
    
    # Perform matching
    matches = match_bibtex_to_arxiv(bib_entries, references, threshold)
    stats['num_matches'] = len(matches)
    
    return matches, stats

def generate_pred_json(
    data_dir: Path,
    output_file: Path,
    num_papers: int = 5,
    threshold: float = 0.7,
    partition: str = 'train'
) -> Dict:
    """
    Generate pred.json file cho một dataset.
    
    Args:
        data_dir: Đường dẫn đến thư mục chứa papers
        output_file: Đường dẫn đến file output
        num_papers: Số papers để xử lý
        threshold: Ngưỡng similarity cho matching
        partition: Partition name ('train', 'val', 'test')
    
    Returns:
        Dict: pred.json structure
    """
    # Get list of paper directories
    paper_dirs = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])[:num_papers]
    
    all_groundtruth = {}
    all_stats = []
    
    print(f"Processing {len(paper_dirs)} papers from {data_dir}...")
    
    for paper_dir in paper_dirs:
        print(f"\n--- Processing: {paper_dir.name} ---")
        matches, stats = process_paper(paper_dir, threshold)
        
        if matches:
            for cite_key, arxiv_id in matches.items():
                # Use paper_id:cite_key as unique identifier
                unique_key = f"{paper_dir.name}:{cite_key}"
                all_groundtruth[unique_key] = arxiv_id
                print(f"  Match: {cite_key} -> {arxiv_id}")
        
        all_stats.append(stats)
        print(f"  Stats: {stats['num_bib_entries']} bib entries, "
              f"{stats['num_references']} refs, {stats['num_matches']} matches")
    
    # Create pred.json structure
    pred_json = {
        "partition": partition,
        "groundtruth": all_groundtruth,
        "prediction": {},  # To be filled by the model later
        "stats": {
            "total_papers": len(paper_dirs),
            "total_matches": len(all_groundtruth),
            "threshold": threshold,
            "per_paper_stats": all_stats
        }
    }
    
    # Save to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved groundtruth to {output_file}")
    print(f"  Total matches: {len(all_groundtruth)}")
    
    return pred_json

def main():
    parser = argparse.ArgumentParser(
        description='Generate groundtruth for Reference Matching Pipeline'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='d:/Coding/KHDL/IntroDS_Milestone2/2210_16298-2211_3000',
        help='Path to data directory containing papers'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='d:/Coding/KHDL/IntroDS_Milestone2/outputs/pred.json',
        help='Output path for pred.json'
    )
    parser.add_argument(
        '--num_papers', 
        type=int, 
        default=5,
        help='Number of papers to process'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.7,
        help='Similarity threshold for matching (0.0-1.0)'
    )
    parser.add_argument(
        '--partition', 
        type=str, 
        default='train',
        choices=['train', 'val', 'test'],
        help='Partition name for the dataset'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_file = Path(args.output)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    generate_pred_json(
        data_dir=data_dir,
        output_file=output_file,
        num_papers=args.num_papers,
        threshold=args.threshold,
        partition=args.partition
    )

if __name__ == '__main__':
    main()
