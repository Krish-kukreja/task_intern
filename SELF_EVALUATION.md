# Kastack Self-Evaluation Sheet

## 1. What went well?
- **Intent Classifier Constraints**: Successfully built a lightweight, completely offline `LinearSVC` + `TF-IDF` intent classification model. The serialized model is under 2MB (well below the 50MB limit) and classifies messages in ~2ms on CPU without relying on external APIs.
- **Adaptive Persona Drift**: The emotional timeline engine successfully detects subtle shifts in user mood (valence, frustration, playfulness) across days and correctly identifies noun-phrase triggers around emotional change points using localized lexicon scoring.
- **Conflict Resolution RAG**: Implemented a resilient retrieval pipeline that detects contradictory factual statements, prioritizing recency and emotional weight to feed clean, merged context to the LLM, reducing hallucinations.
- **Sync Architecture Design**: Outlined a robust, privacy-first Event-Sourced CRDT sync model that prevents destructive overwrites (unlike Last-Write-Wins), perfectly tailored for append-mostly chat environments.

## 2. What was challenging?
- **Topic Splitting on Messy Data**: Detecting meaningful triggers for persona drift was difficult because the dataset lacks continuous timestamps (only sequential `day` integers). We had to rely strictly on sequential distance and localized noun-chunking to infer causality.
- **Lightweight Offline Constraint**: Balancing accuracy and the <50MB model limit for intent classification required moving away from heavy Transformer models and relying on classical NLP (TF-IDF + SVM) coupled with a strong training dataset.
- **LFS Deployment to Hugging Face**: Bypassing Hugging Face's 10MB file limit for our embedding index and intent model required careful Git LFS history rewriting, ensuring large binaries were tracked without breaking the CI pipeline.

## 3. What would you improve with more time?
- **Vector Clocks / CRDT Implementation**: While we designed the architecture theoretically, implementing the actual Hybrid Logical Clocks (HLC) and tombstoning logic on the client side would be the next practical step.
- **Better Entity Resolution**: In the conflict-resolution RAG, mapping entities (e.g., "sister" -> "sibling") using a lightweight local knowledge graph rather than raw keyword matching would make contradiction detection more semantic and resilient to phrasing variations.
- **Real-time UX**: Streamlining the frontend to show the "persona drift timeline" visually as the user chats, rather than just in the Insights dashboard.

## 4. Architectural Trade-offs Made
- **Lexicon vs. LLM for Emotion**: We chose a lexicon-based scorer for the persona drift engine instead of an LLM. While an LLM might catch more nuance, the lexicon runs instantly on-device, preserving privacy and keeping compute costs strictly at zero.
- **TF-IDF vs DistilBERT**: We opted for a classical TF-IDF SVM for the intent classifier. A DistilBERT model would offer slightly better zero-shot generalizability, but would push the memory boundaries (often >200MB) and latency constraints, violating the core <50MB prompt requirement.
