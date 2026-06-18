"""
rag_engine.py — Retrieval-Augmented Generation Engine

Handles:
  1. Query embedding
  2. Message-level retrieval (cosine similarity over all 191K embeddings)
  3. Topic-level retrieval (precomputed centroids)
  4. Checkpoint retrieval
  5. Answer generation via flan-t5-small
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
EMB_PATH       = BASE_DIR / "data" / "processed" / "embeddings.npy"
INDEX_PATH     = BASE_DIR / "data" / "processed" / "embedding_index.json"
JSONL_PATH     = BASE_DIR / "data" / "processed" / "processed_messages.jsonl"
TOPICS_PATH    = BASE_DIR / "data" / "processed" / "topic_segments.json"
SUMMARIES_PATH = BASE_DIR / "data" / "processed" / "summaries.json"
PERSONA_PATH   = BASE_DIR / "data" / "processed" / "persona.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
QA_MODEL    = "google/flan-t5-small"


class RAGEngine:
    """Lazy-loaded RAG engine. Call .initialize() once at startup."""

    def __init__(self):
        self.ready = False
        self.embedder = None
        self.qa_model = None
        self.embeddings = None          # shape [N, 384], already L2-normalized
        self.msg_index = None           # list[int]: array_pos -> msg_id
        self.messages = {}              # msg_id -> record dict
        self.topics = []                # list of topic segment dicts
        self.topic_centroids = None     # shape [T, 384]
        self.checkpoints = []           # list of checkpoint dicts
        self.persona = {}

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self):
        """Load all assets into memory. Call once at server startup."""
        import time
        t0 = time.time()
        print("[RAG] Loading sentence-transformer...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("[RAG] Loading QA model (flan-t5-small)...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL)

        print("[RAG] Loading embeddings...")
        raw = np.load(EMB_PATH).astype(np.float32)
        # Ensure L2-normalized
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.embeddings = raw / np.maximum(norms, 1e-10)

        print("[RAG] Loading message index...")
        with open(INDEX_PATH, 'r') as f:
            idx = json.load(f)
        self.msg_index = [int(idx[str(i)]) for i in range(len(idx))]
        self.mid_to_index = {mid: i for i, mid in enumerate(self.msg_index)}

        print("[RAG] Loading messages...")
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                self.messages[rec["msg_id"]] = rec

        print("[RAG] Loading topics...")
        with open(TOPICS_PATH, 'r', encoding='utf-8') as f:
            self.topics = json.load(f)

        print("[RAG] Precomputing topic centroids...")
        self._precompute_centroids()

        print("[RAG] Loading summaries...")
        self._load_summaries()

        print("[RAG] Loading persona...")
        with open(PERSONA_PATH, 'r', encoding='utf-8') as f:
            self.persona = json.load(f)

        elapsed = time.time() - t0
        print(f"[RAG] Ready in {elapsed:.1f}s")
        self.ready = True

    def _precompute_centroids(self):
        """Precompute L2-normalized centroid for each topic segment."""
        centroids = []
        for topic in self.topics:
            start_id = topic["start_msg_id"]
            end_id   = topic["end_msg_id"]
            
            # Fast lookup
            indices = [self.mid_to_index[mid] for mid in range(start_id, end_id + 1) if mid in self.mid_to_index]
            if indices:
                chunk = self.embeddings[indices]
                centroid = chunk.mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids.append(centroid / max(norm, 1e-10))
            else:
                centroids.append(np.zeros(384, dtype=np.float32))
        self.topic_centroids = np.array(centroids, dtype=np.float32)

    def _load_summaries(self):
        """Load checkpoint summaries from summaries.json (may not exist yet)."""
        if SUMMARIES_PATH.exists():
            with open(SUMMARIES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.checkpoints = data.get("checkpoint_summaries", [])
            # Also update topic summaries if available
            topic_sum_map = {ts["topic_id"]: ts["summary"]
                             for ts in data.get("topic_summaries", [])}
            for t in self.topics:
                if t["topic_id"] in topic_sum_map:
                    t["summary"] = topic_sum_map[t["topic_id"]]
            print(f"[RAG]   Loaded {len(self.checkpoints)} checkpoint summaries")

            print("[RAG]   Embedding topic summaries...")
            topic_summaries_text = [t.get("summary", "") for t in self.topics]
            topic_embs = self.embedder.encode(topic_summaries_text, convert_to_numpy=True)
            t_norms = np.linalg.norm(topic_embs, axis=1, keepdims=True)
            self.topic_summary_embeddings = topic_embs / np.maximum(t_norms, 1e-10)

            print("[RAG]   Embedding checkpoint summaries...")
            ckpt_summaries_text = [ck.get("summary", "") for ck in self.checkpoints]
            if ckpt_summaries_text:
                ck_embs = self.embedder.encode(ckpt_summaries_text, convert_to_numpy=True)
                ck_norms = np.linalg.norm(ck_embs, axis=1, keepdims=True)
                self.checkpoint_summary_embeddings = ck_embs / np.maximum(ck_norms, 1e-10)
            else:
                self.checkpoint_summary_embeddings = np.array([])
        else:
            print("[RAG]   summaries.json not found — checkpoint retrieval will be empty")
            self.checkpoints = []
            self.topic_summary_embeddings = None
            self.checkpoint_summary_embeddings = None

    # ── Query embedding ───────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Embed and L2-normalize a query string."""
        # Pre-process the query to remove exact persona name bias.
        # Since retrieval already filters by target_user, asking "what is user 1 job"
        # biases the embedding heavily towards "Hi, I'm user 1". We rewrite to "you/your".
        processed = query.lower()
        processed = processed.replace("user 1's", "your")
        processed = processed.replace("user 2's", "your")
        processed = processed.replace("user 1", "you")
        processed = processed.replace("user 2", "you")
        
        vec = self.embedder.encode([processed], convert_to_numpy=True)[0]
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-10)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_relevant_messages(self, query_emb: np.ndarray, target_user: str = None, target_topic: str = None, top_k: int = 10) -> list[dict]:
        """Top-k message retrieval via cosine similarity."""
        
        start_id = 0
        end_id = float('inf')
        if target_topic:
            for t in self.topics:
                if str(t["topic_id"]) == str(target_topic):
                    start_id = t.get("start_msg_id", 0)
                    end_id = t.get("end_msg_id", float('inf'))
                    break
                    
        if target_user or target_topic:
            indices = []
            for i, mid in enumerate(self.msg_index):
                if not (start_id <= mid <= end_id):
                    continue
                if target_user and self.messages.get(mid, {}).get("sender") != target_user:
                    continue
                indices.append(i)
                
            if not indices:
                return []
            embs = self.embeddings[indices]
            sims = embs @ query_emb
        else:
            indices = list(range(len(self.msg_index)))
            sims = self.embeddings @ query_emb

        k = min(top_k, len(sims))
        if k == 0:
            return []

        top_local_idx = np.argpartition(sims, -k)[-k:]
        top_local_idx = top_local_idx[np.argsort(sims[top_local_idx])[::-1]]

        results = []
        for loc_idx in top_local_idx:
            orig_idx = indices[loc_idx]
            mid = self.msg_index[orig_idx]
            rec = self.messages.get(mid, {})
            results.append({
                "msg_id": mid,
                "text": rec.get("message_text", ""),
                "sender": rec.get("sender", ""),
                "day": rec.get("day", 0),
                "conversation_id": rec.get("conversation_id", 0),
                "similarity_score": float(sims[loc_idx])
            })
        return results

    def retrieve_relevant_topics(self, query_emb: np.ndarray, top_k: int = 3) -> list[dict]:
        """Top-k topic retrieval via summary embedding cosine similarity."""
        if not hasattr(self, 'topic_summary_embeddings') or self.topic_summary_embeddings is None or len(self.topic_summary_embeddings) == 0:
            return []

        sims = self.topic_summary_embeddings @ query_emb     # shape [T]
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            t = self.topics[idx]
            results.append({
                "topic_id": t["topic_id"],
                "start_msg_id": t["start_msg_id"],
                "end_msg_id": t["end_msg_id"],
                "start_day": t["start_day"],
                "end_day": t["end_day"],
                "msg_range": f"{t['start_msg_id']}-{t['end_msg_id']}",
                "num_messages": t["num_messages"],
                "summary": t.get("summary", ""),
                "similarity_score": float(sims[idx])
            })
        return results

    def retrieve_relevant_checkpoints(self, query_emb: np.ndarray, top_k: int = 2) -> list[dict]:
        """Top-k checkpoint retrieval via summary embedding cosine similarity."""
        if not hasattr(self, 'checkpoint_summary_embeddings') or self.checkpoint_summary_embeddings is None or len(self.checkpoint_summary_embeddings) == 0:
            return []

        sims = self.checkpoint_summary_embeddings @ query_emb
        k = min(top_k, len(sims))
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            ck = self.checkpoints[idx]
            results.append({
                "checkpoint_id": ck["checkpoint_id"],
                "msg_range": ck["msg_range"],
                "start_day": ck["start_day"],
                "end_day": ck["end_day"],
                "summary": ck.get("summary", ""),
                "similarity_score": float(sims[idx])
            })
        return results

    # ── Answer generation ──────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        messages: list[dict],
        topics: list[dict],
        checkpoints: list[dict]
    ) -> str:
        """Generate a grounded answer using flan-t5-small with strict anti-echo prompt."""

        context_str = "Topic Summaries:\n"
        for t in topics:
            context_str += f"- {t.get('summary', 'N/A')}\n"
            
        context_str += "\nCheckpoint Summaries:\n"
        for c in checkpoints:
            context_str += f"- {c.get('summary', 'N/A')}\n"
            
        context_str += "\nRelevant Messages:\n"
        for m in messages:
            context_str += f"- {m['sender']}: {m['text']}\n"
            
        prompt = f"""Based ONLY on the context below, answer the user's question. 
        Do not repeat the context. Synthesize a natural answer. 
        If the context doesn't contain the answer, say 'I don't know based on the provided context.'
        
        Context:
        {context_str}
        
        Question: {query}
        Answer:"""

        try:
            inputs = self.qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.qa_model.generate(**inputs, max_new_tokens=150, do_sample=False)
            ans = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return ans
        except Exception as e:
            return f"[Generation error: {e}]"

    # ── Main query interface ──────────────────────────────────────────────────

    def query(self, user_query: str, target_user: str = None, target_topic: str = None, top_k_msgs: int = 10,
              top_k_topics: int = 3, top_k_checkpoints: int = 2) -> dict:
        """Full RAG pipeline: embed → retrieve → generate → return."""
        if not self.ready:
            return {"error": "RAG engine not initialized"}

        query_emb = self.embed_query(user_query)

        msgs       = self.retrieve_relevant_messages(query_emb, target_user=target_user, target_topic=target_topic, top_k=top_k_msgs)
        topics     = self.retrieve_relevant_topics(query_emb, top_k=top_k_topics)
        checkpoints = self.retrieve_relevant_checkpoints(query_emb, top_k=top_k_checkpoints)

        max_sim = msgs[0]["similarity_score"] if msgs else 0
        if max_sim < 0.30:
            answer = "I couldn't find any highly relevant messages to answer this question accurately. Please try rephrasing or ask about a different topic."
            no_results = True
        else:
            if is_persona_query(user_query) and msgs:
                answer = format_persona_answer(user_query, self.persona)
            else:
                answer = self.generate_answer(user_query, msgs, topics, checkpoints)
            no_results = False

        return {
            "query": user_query,
            "answer": answer,
            "no_results": no_results,
            "sources": {
                "topics_used": [
                    {
                        "id": t["topic_id"],
                        "range": t["msg_range"],
                        "summary": t["summary"],
                        "score": t["similarity_score"]
                    } for t in topics
                ],
                "checkpoints_used": [
                    {
                        "id": ck["checkpoint_id"],
                        "range": ck["msg_range"],
                        "summary": ck["summary"],
                        "score": ck["similarity_score"]
                    } for ck in checkpoints
                ],
                "messages_used": [
                    {
                        "msg_id": m["msg_id"],
                        "text": m["text"][:200],
                        "sender": m["sender"],
                        "day": m["day"],
                        "score": m["similarity_score"]
                    } for m in msgs[:5]
                ]
            }
        }


# ── Persona query helpers ─────────────────────────────────────────────────────

PERSONA_KEYWORDS = {
    "habit": ["habit", "routine", "sleep", "wake", "morning", "night"],
    "talk": ["talk", "speak", "communicate", "message", "write", "style"],
    "person": ["person", "personality", "character", "like", "who", "kind of"],
    "job": ["job", "work", "career", "profession", "do for"],
    "location": ["live", "from", "location", "city", "country", "where"],
    "relationship": ["relationship", "family", "friend", "partner", "boyfriend", "girlfriend"],
    "hobby": ["hobby", "interest", "enjoy", "fun", "leisure"],
}


def is_persona_query(query: str) -> bool:
    """Detect if query is specifically about persona/traits vs conversation content."""
    q = query.lower()
    # Strong persona signals — these clearly ask about WHO the person is
    strong_triggers = [
        "what kind of person",
        "personality",
        "habits",
        "how do they talk",
        "how do they communicate",
        "how do they speak",
        "communication style",
        "describe their personality",
        "describe the person",
        "user 1 like",
        "user 2 like",
        "what are they like",
        "profile",
        "tell me about their habits",
        "tell me about their personality",
        "tell me about their communication",
        "live",
        "location",
        "hobbies",
        "hobby",
        "who is user",
        "who was user",
        "tell me about user"
    ]
    return any(trigger in q for trigger in strong_triggers)


def format_persona_answer(query: str, global_persona: dict) -> str:
    """Format global persona data into a natural-language answer."""
    q = query.lower()
    u1 = global_persona.get("persona_user_1", {})
    u2 = global_persona.get("persona_user_2", {})

    lines = ["Based on an analysis of all 11,000+ conversation threads:\n"]

    # Detect what aspect they're asking about
    if any(w in q for w in ["habit", "sleep", "routine", "late", "early"]):
        lines.append("**Habits:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            h = u.get("habits", {})
            traits = []
            if isinstance(h.get("late_sleeper"), dict) and h["late_sleeper"].get("detected"):
                traits.append(f"late sleeper ({h['late_sleeper']['evidence_count']} content mentions)")
            if isinstance(h.get("early_bird"), dict) and h["early_bird"].get("detected"):
                traits.append(f"early bird ({h['early_bird']['evidence_count']} content mentions)")
            if h.get("brief_communicator"):
                traits.append("brief communicator")
            if h.get("verbose_communicator"):
                traits.append("verbose communicator")
            lines.append(f"  {label}: {', '.join(traits) if traits else 'no distinct habits detected'}")

    elif any(w in q for w in ["talk", "speak", "communicate", "style", "message"]):
        lines.append("**Communication Style:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            s = u.get("communication_style", {})
            lines.append(f"  {label}:")
            lines.append(f"    • Avg message length: {s.get('avg_message_length', 0)} chars")
            lines.append(f"    • Uses exclamations: {s.get('exclamation_rate', 0)*100:.0f}% of messages")
            lines.append(f"    • Asks questions: {s.get('question_rate', 0)*100:.0f}% of messages")
            lines.append(f"    • Emoji usage: {s.get('emoji_usage_rate', 0)*100:.1f}%")

    elif any(w in q for w in ["job", "work", "career", "profession"]):
        lines.append("**Most commonly mentioned jobs:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            jobs = u.get("personal_facts", {}).get("job_mentions", {})
            filtered = {k: v for k, v in jobs.items() if not any(stop in k for stop in ["glad", "sorry", "sure", "same", "free", "relax"])}
            top = list(filtered.items())[:5]
            lines.append(f"  {label}: {', '.join(f'{j} ({c})' for j,c in top) if top else 'none mentioned'}")

    elif any(w in q for w in ["personality", "person", "character", "kind of"]):
        lines.append("**Personality Traits:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            t = u.get("personality_traits", {})
            detected = [name for name, val in t.items()
                        if isinstance(val, dict) and val.get("detected")]
            lines.append(f"  {label}: {', '.join(detected) if detected else 'neutral/mixed'}")

    elif any(w in q for w in ["location", "live", "from", "where"]):
        lines.append("**Commonly mentioned locations:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            locs = u.get("personal_facts", {}).get("location_mentions", {})
            top = list(locs.items())[:5]
            lines.append(f"  {label}: {', '.join(f'{l} ({c})' for l,c in top) if top else 'none mentioned'}")

    else:
        # General overview
        lines.append("**Overview of both participants:**")
        for label, u in [("User 1", u1), ("User 2", u2)]:
            if not u: continue
            s = u.get("communication_style", {})
            t = u.get("personality_traits", {})
            detected_traits = [name for name, val in t.items()
                                if isinstance(val, dict) and val.get("detected")]
            jobs = list(u.get("personal_facts", {}).get("job_mentions", {}).keys())[:3]
            filtered_jobs = [j for j in jobs if not any(stop in j for stop in ["glad", "sorry", "sure", "same", "free", "relax"])]
            lines.append(f"  {label} ({u.get('total_messages_analyzed', 0):,} messages):")
            lines.append(f"    - Traits: {', '.join(detected_traits) or 'neutral'}")
            lines.append(f"    - Avg message: {s.get('avg_message_length', 0)} chars")
            lines.append(f"    - Top jobs mentioned: {', '.join(filtered_jobs) if filtered_jobs else 'none'}")

    return "\n".join(lines)
