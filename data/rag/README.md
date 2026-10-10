# Anonymized RAG knowledge base

This directory contains the knowledge records used by the reported hybrid-RAG workflow.

- `rag_knowledge_base.xlsx`: auditable structured knowledge table.
- `rag_corpus.jsonl`: retrieval corpus, with one knowledge chunk per line.
- `well_enterprise_production_candidates.jsonl`: anonymized well-enterprise candidate records.
- `spatial_anonymized/`: shifted well coordinates, shifted enterprise boundaries, and an anonymization note.

The corpus covers chemical-indicator interpretation, confirmatory analytical methods, EEM and UV-Vis feature interpretation, risk-label mapping, experimental conditions, hydrogeological context, and anonymized enterprise production information. Knowledge entries retain source identifiers and applicability limits.

Enterprise names and original coordinates are not included. Enterprises use `E001-E060`; the 41 modeled wells use the anonymized `CK`, `A`, `B`, `C`, and `D` identifiers, and additional historical well records use `H` identifiers. A common coordinate shift preserves relative distances, directions, point-in-polygon relationships, and regional hydraulic comparisons.

Independent source-attribution references are not stored in the retrieval corpus. They are held separately under `data/llm/` and are opened only during output evaluation.
