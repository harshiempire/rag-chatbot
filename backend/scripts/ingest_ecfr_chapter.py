import argparse
import json
import os
import sys
import uuid
import requests
import xml.etree.ElementTree as ET
import psycopg2
from dotenv import load_dotenv


def find_chapter(title_json, chapter_identifier):
    for child in title_json.get('children', []):
        if child.get('type') == 'chapter' and child.get('identifier') == chapter_identifier:
            return child
    return None


def collect_parts(node):
    parts = []
    def _walk(n):
        if isinstance(n, dict):
            if n.get('type') == 'part':
                if n.get('identifier'):
                    parts.append(n.get('identifier'))
            for ch in n.get('children', []) or []:
                _walk(ch)
    _walk(node)
    # Preserve order but remove duplicates
    seen = set()
    ordered = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def iter_sections_from_part(title, date, part):
    url = f'https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}'
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    xml_root = ET.fromstring(resp.text)
    for div in xml_root.iter():
        if div.tag.startswith('DIV') and div.attrib.get('TYPE') == 'SECTION':
            head = div.find('HEAD')
            text_parts = []
            if head is not None and head.text:
                text_parts.append(head.text.strip())
            for p in div.findall('P'):
                p_text = ''.join(p.itertext()).strip()
                if p_text:
                    text_parts.append(p_text)
            content = '\n'.join(text_parts).strip()
            if not content:
                continue
            yield {
                'section': div.attrib.get('N', ''),
                'heading': head.text.strip() if head is not None and head.text else '',
                'content': content,
            }


def main():
    parser = argparse.ArgumentParser(description='Ingest eCFR chapter into PostgreSQL')
    parser.add_argument('--title', type=int, required=True, help='Title number (e.g., 12)')
    parser.add_argument('--chapter', type=str, required=True, help='Chapter identifier (e.g., XII)')
    parser.add_argument('--date', type=str, default='2024-01-01', help='eCFR version date or "current"')
    parser.add_argument('--source-id', type=str, default=None, help='Source id tag for documents')
    parser.add_argument('--batch-size', type=int, default=10, help='How many parts to ingest in one run')
    parser.add_argument('--batch-index', type=int, default=0, help='Batch index (0-based)')
    args = parser.parse_args()

    load_dotenv(dotenv_path='.env')
    conn_str = os.getenv('DATABASE_URL')
    if not conn_str:
        print('DATABASE_URL not set', file=sys.stderr)
        return 1

    structure_url = f'https://www.ecfr.gov/api/versioner/v1/structure/{args.date}/title-{args.title}.json'
    resp = requests.get(structure_url, timeout=60)
    resp.raise_for_status()
    title_json = resp.json()

    chapter = find_chapter(title_json, args.chapter)
    if not chapter:
        print(f'Chapter {args.chapter} not found', file=sys.stderr)
        return 1

    parts = collect_parts(chapter)
    if not parts:
        print('No parts found for chapter', file=sys.stderr)
        return 1

    total_parts = len(parts)
    start = args.batch_index * args.batch_size
    end = start + args.batch_size
    batch_parts = parts[start:end]

    if not batch_parts:
        print('No parts in this batch', file=sys.stderr)
        return 1

    source_id = args.source_id or f'ecfr-title-{args.title}-chapter-{args.chapter.lower()}'

    inserted = 0
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for part in batch_parts:
                for section in iter_sections_from_part(args.title, args.date, part):
                    doc_id = str(uuid.uuid4())
                    metadata = {
                        'title': args.title,
                        'chapter': args.chapter,
                        'part': part,
                        'section': section['section'],
                        'heading': section['heading'],
                        'type': 'SECTION',
                        'source': 'ecfr',
                        'date': args.date
                    }
                    cur.execute('''
                        INSERT INTO documents (id, source_id, content, metadata, classification, status, pii_detected)
                        VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                    ''', (
                        doc_id,
                        source_id,
                        section['content'],
                        json.dumps(metadata),
                        'public',
                        'published',
                        False
                    ))
                    inserted += 1
        conn.commit()

    print(f'Total parts: {total_parts}')
    print(f'Batch parts: {batch_parts}')
    print(f'Inserted sections: {inserted}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
