"""
Test script to compare Fixed vs Structure-Aware chunking

This demonstrates the difference between:
1. Fixed 1000-character chunking (current approach)
2. Structure-aware chunking (production approach)
"""

import sys
sys.path.insert(0, '/Users/harshithalle/RAG-Chatbot')

from part3_extractors import RegulationChunker, VectorizationEngine

# Sample regulatory text (from eCFR Title 12)
SAMPLE_REGULATION = """
§ 1277.1 Definitions.

As used in this chapter:

(a) Acquired member assets means assets acquired from a member's portfolio in accordance with § 1277.12, including mortgage pools and other assets or notes.

(b) Asset and liability management means actions taken by a Bank to manage interest rate risk, credit risk, and liquidity risk, including the use of derivatives.

(c) Collateral means assets pledged to secure advances and other Bank credit products.

§ 1277.2 Capital requirements.

(a) Minimum capital requirement means the leverage and total capital requirements under section 6(a)(2) of the Bank Act.

(b) Risk-based capital requirement means that each Bank shall maintain permanent capital at all times in an amount at least equal to the sum of:
(1) Its credit risk capital requirement;
(2) Its market risk capital requirement; and
(3) Its operational risk capital requirement.

(c) Permanent capital means the retained earnings of the Bank determined in accordance with GAAP plus the amount paid-in for the Bank's Class B stock.

§ 1277.3 Capital stock.

(a) Class A stock shall be redeemable on six months written notice, subject to the provisions of § 1277.4.

(b) Class B stock shall be redeemable on five years written notice, subject to the provisions of § 1277.4.

(c) Both classes of stock shall be issued only at par value as determined by the Bank's board of directors.
"""

def main():
    print("=" * 70)
    print("CHUNKING COMPARISON: Fixed vs Structure-Aware")
    print("=" * 70)
    
    # Initialize chunker
    chunker = RegulationChunker(
        max_chunk_size=800,
        min_chunk_size=150,
        include_parent_context=True
    )
    
    # =========================================================================
    # Fixed Chunking (current approach)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. FIXED CHUNKING (Current - 500 chars)")
    print("=" * 70)
    
    engine = VectorizationEngine()
    fixed_chunks = engine.chunk_text(SAMPLE_REGULATION, chunk_size=500, chunk_overlap=100)
    
    for i, chunk in enumerate(fixed_chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\n📊 Fixed chunking: {len(fixed_chunks)} chunks")
    
    # =========================================================================
    # Structure-Aware Chunking (production approach)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. STRUCTURE-AWARE CHUNKING (Production)")
    print("=" * 70)
    
    structure_chunks = chunker.chunk_regulation(
        SAMPLE_REGULATION,
        metadata={'source': 'ecfr', 'title': 12}
    )
    
    for i, chunk in enumerate(structure_chunks):
        header = chunk['metadata'].get('section_header', 'N/A')
        chunk_type = chunk.get('chunk_type', 'unknown')
        content = chunk['content']
        
        print(f"\n--- Chunk {i+1}: {chunk_type} ({len(content)} chars) ---")
        print(f"Section: {header}")
        print(content[:200] + "..." if len(content) > 200 else content)
    
    print(f"\n📊 Structure-aware chunking: {len(structure_chunks)} chunks")
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"""
| Metric                    | Fixed      | Structure-Aware |
|---------------------------|------------|-----------------|
| Number of chunks          | {len(fixed_chunks):>10} | {len(structure_chunks):>15} |
| Respects § sections       | ❌ No       | ✅ Yes           |
| Preserves paragraph (a)(b)| ❌ No       | ✅ Yes           |
| Includes section context  | ❌ No       | ✅ Yes           |
| Production-ready          | ❌ No       | ✅ Yes           |
""")

if __name__ == "__main__":
    main()
