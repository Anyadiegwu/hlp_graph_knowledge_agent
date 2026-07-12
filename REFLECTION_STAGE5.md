# REFLECTION_STAGE5.md — HLP Graph Knowledge Agent

## 1. The Vulnerabilities of Distance Threshold Tuning

`TrackedRedisSemanticCache` short-circuits the sampling model behind a single
scalar: `distance_threshold`. That number is doing more architectural work than
its size suggests — it is the entire boundary between "this is the same
question" and "this is a different question," decided by cosine distance
between two embedding vectors rather than by anything resembling understanding.

Tuning it too loose (say, 0.4–0.5) treats semantically adjacent-but-distinct
prompts as identical. "What caused the timeout in session A?" and "What caused
the timeout in session B?" can sit close together in embedding space purely
because they share vocabulary and structure, even though the correct answers
are session-specific and mutually exclusive. A loose threshold returns session
A's cached diagnosis for session B's question — a false-positive hit that is
worse than a cache miss, because it actively injects a wrong but confident-sounding
answer into the agent's reasoning loop. The token savings are real, but they are
being purchased with silent correctness failures that are hard to detect after
the fact, since the response still reads as fluent and plausible.

Tuning it too tight (0.05 or below) makes the cache nearly inert. Any rewording,
added qualifier, or reordered clause pushes the embedding distance just past the
cutoff, and every request falls through to a full LLM round-trip anyway. The
system pays the full latency and token cost of an uncached architecture while
carrying the additional complexity and failure surface of a cache that is
almost never actually consulted.

We settled on 0.15 as a middle position, but it is an aggregate, static number
applied uniformly across every kind of sampling request the agent makes —
factual lookups, error diagnoses, and open-ended reasoning prompts alike. That
uniformity is itself a limitation: a diagnostic question about a specific trace
ID arguably deserves a much tighter threshold (near-zero tolerance for returning
another trace's answer) than a general knowledge-style sampling request, where
a looser match is comparatively low-risk. A more mature version of this system
would tier the threshold by request type rather than applying one number
globally, and would log near-miss distances (requests that almost matched) to
make retroactive auditing of threshold correctness possible.

---

## 2. State Synchronization Challenges in Cloud-Native Multi-Agent Networks

Moving `HLPLogStore` from a local SQLite file to a pooled `asyncpg` connection
against Supabase traded one failure mode for a different, less forgiving one.
SQLite's failure mode was almost entirely local: a locked file, a full disk. The
Supabase failure mode is a live network dependency — every `put()` and `search()`
call now crosses the public internet to a managed Postgres instance, and every
one of the resilience assumptions built for a single-process local file no
longer holds.

The connection pool (`asyncpg.create_pool(min_size=2, max_size=10, ...)`) is the
first place this shows up concretely. A pool bounds concurrency, but it does not
make concurrent writers coordinate with each other. `_get_or_create_session_row`
uses `INSERT ... ON CONFLICT DO UPDATE` specifically because two coroutines
racing to create the same session row would otherwise raise a unique-constraint
violation — a problem that simply could not occur with a single-threaded SQLite
connection guarded by a Python-level lock. Distributed state means every write
path that used to be implicitly serialized now has to be made explicitly
idempotent.

The second hazard is partial connectivity. A dropped connection mid-transaction,
a pooler timeout under load, or a transient DNS failure on Supabase's side can
leave a `log_entries` insert half-completed relative to its `sessions` row, in
a way that never had an analog when everything lived in one file with one
writer. `asyncpg`'s pool will retry acquiring a *connection*, but it does not
retry a *failed query* — an exception from a dropped connection still propagates
up through `put()`, which is exactly why every log-write call site is wrapped in
its own `try/except` rather than assuming the write always lands.

The third hazard is consistency across the two now-independent Postgres
consumers: `agent_client`'s writer pool and `analysis_dashboard`'s reader pool
(`SharedLogStoreReader`). These are separate pools in separate processes,
possibly on separate machines. Supabase's Postgres gives us strong consistency
*within* a single query, but nothing enforces that the dashboard is reading a
state that reflects the agent's most recent write — there is a real, if narrow,
window where a session exists in the writer's pool's view of the world
fractionally before a concurrent reader's pool observes it, purely as a function
of connection-level caching and network latency, not application logic.

None of this is a reason to prefer SQLite — a single local file cannot serve
concurrent multi-agent execution at all, which was the entire premise of this
migration. It is a reason to treat "the database call might simply not happen"
as a first-class case to design for, in a way that a synchronous local file
write never required.

---

## 3. Local Volatile Storage vs. Centralized Persistent Caching

The clearest way to see the difference is to ask what happens to the cache when
the process restarts. A plain Python dictionary living inside `agent_client`'s
process — the kind of in-loop memoization you'd reach for by default — dies the
instant that process exits. It is fast (nanoseconds, no serialization, no
network hop) precisely because it never leaves the process's own memory, but
that speed is inseparable from its scope: only the coroutine that wrote to it
can ever read from it.

Redis breaks that coupling on purpose. `TrackedRedisSemanticCache`,
`_tier2_get_or_compute`, and `_tier3_get_or_compute` all write to the *same*
external Redis instance from three different processes — `agent_client`'s
sampling handler, `analysis_dashboard`'s LangGraph agent, and the Streamlit UI
— none of which share a Python process, and none of which could see each
other's data if the cache were a local dict. The `hlp:cache:tier1:hits` counter
that the dashboard displays is only meaningful *because* it was incremented by
a completely separate process talking to the agent client; a local dictionary
could never produce that number.

This is the real structural tradeoff, and it runs in both directions. The
in-memory dictionary wins on raw latency and simplicity for anything that is
genuinely single-process and doesn't need to survive a restart — there is no
reason to pay a network round-trip to Redis for a value that only ever matters
within one function call's lifetime. Redis wins the moment more than one running
instance needs to see the same cache state, which is exactly the situation this
system is in: nothing about the multi-tier architecture in this stage would
work if agent_client, analysis_dashboard, and the Streamlit UI could not read
each other's cache writes.

The scaling implication follows directly from that. If `agent_client` were ever
horizontally scaled — multiple instances of the sampling handler running behind
a queue, say — an in-process dictionary cache would mean each instance builds
and holds its own private, inconsistent copy of "what's already been answered,"
with cache hit rates that degrade as more instances are added, since a warm
entry in instance A's memory is invisible to instance B. Because the cache
already lives in Redis instead, adding more instances of `agent_client` costs
nothing architecturally: every instance shares one hit rate, one token-savings
counter, one semantic index, regardless of how many processes are pointed at
it. The centralized cache is what makes horizontal scaling free rather than
something that requires re-architecting the caching layer later.