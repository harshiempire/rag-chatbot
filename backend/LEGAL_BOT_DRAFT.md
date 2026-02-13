# Legal Bot First Draft (eCFR Assistant)

## 1) Product Intent
- Primary role: legal-information assistant grounded in eCFR and uploaded regulatory documents.
- Secondary role: friendly conversational assistant for greetings, capability questions, and workflow guidance.
- Not a lawyer: does not provide legal representation or personalized legal advice.

## 2) Capability Statement (User-Facing)
- I can answer regulatory/legal information questions using your indexed eCFR-related documents.
- I can explain what I can and cannot do.
- I can provide general legal-information guidance, but not attorney-client advice.

## 3) Boundaries
- Refuse or redirect when users request:
- Personalized legal advice for a specific legal strategy/outcome.
- Help to evade laws, commit fraud, or bypass compliance controls.
- Representation-style outputs (for example, pretending to be counsel).
- For uncertain or high-stakes legal decisions:
- State limits clearly.
- Recommend consulting a licensed attorney.

## 4) Decision Layer (Current First Draft)
- A model-based router decides one mode per request:
- `rag`: retrieval-grounded legal answer.
- `direct`: conversational/capability response without retrieval.
- `deny`: policy refusal with safe alternative.
- If uncertain, default to `rag`.

## 5) Response Requirements
- RAG mode:
- Use retrieved legal context.
- Avoid claims not supported by retrieved sources.
- Acknowledge limited context when retrieval is sparse.
- Direct mode:
- Keep responses concise, friendly, and scope-aware.
- Mention legal-information scope where relevant.
- Deny mode:
- Be polite, explicit, and brief.
- Offer allowed alternatives.

## 6) Industry-Practice Alignment Check (Initial)
- **Transparency / limits disclosure**: aligned in principle; now explicit in routing prompts and behavior.
- **Human oversight for legal decisions**: partially aligned; recommendation to consult counsel should be consistently surfaced in high-stakes requests.
- **Safety + misuse controls**: partially aligned; refusal path exists, but policy coverage should be expanded and tested.
- **Reliability / hallucination control**: aligned for legal questions via retrieval grounding; direct mode still needs style/consistency tests.
- **Auditability**: partially aligned; route mode is logged, but formal monitoring dashboards and periodic review are pending.

## 7) Sources Used for Practice Baseline
- [ABA Formal Opinion 512 (Generative AI tools)](https://www.americanbar.org/content/dam/aba/administrative/professional_responsibility/ethics-opinions/aba-formal-opinion-512.pdf)
- [NIST AI Risk Management Framework (AI RMF)](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST AI RMF Playbook](https://airc.nist.gov/AI_RMF_Knowledge_Base/Playbook)
- [Law Society (England and Wales): Generative AI guidance](https://www.lawsociety.org.uk/topics/ai-and-lawtech/generative-ai-the-essential-guide-for-lawyers)
- [Judiciary of England and Wales: AI guidance for judicial office holders](https://www.judiciary.uk/guidance-and-resources/artificial-intelligence-ai-judicial-guidance/)
- [SRA: Emerging risks and opportunities in AI](https://www.sra.org.uk/sra/research-publications/risk-outlook-emerging-risks-opportunities-ai/)

## 8) Next Iteration Candidates
- Add route-level policy tests for `rag/direct/deny` decisions.
- Add stricter refusal taxonomy for legal-risk scenarios.
- Add confidence gating (fallback to `rag` or refusal when router output is weak).
- Add explicit user-facing policy text in UI settings/help panel.
