# REFLECTION_STAGE4.md — HLP Graph Knowledge Agent

## 1. Deterministic vs. Edgeless Graph Orchestration

Traditional LangGraph topologies use `builder.add_edge()` to declare every
transition at compile time. The graph structure is fixed, fully inspectable,
and trivially visualisable — you can call `graph.get_graph().draw_mermaid()` and
see every possible path before a single token is generated. Testing is
straightforward: you know exactly which nodes will run for a given input because
the routing is structural, not behavioural.

Discarding explicit edges in favour of `Command(goto=...)` objects shifts routing
responsibility into the nodes themselves. Each node reads the current state,
applies its own logic, and decides dynamically where execution goes next. This
produces genuinely adaptive pipelines — the same compiled graph can execute
`initial_ingest → stats → xai → chart → synthesize` for an audit query and
`initial_ingest → stats → synthesize` for a simple statistics question, without
any conditional branching defined outside the nodes.

The operational hazards are real. Without hardcoded edges, static analysis tools
cannot enumerate all possible execution paths. A bug in a node's routing logic —
returning a `goto` that names a non-existent node, or entering a cycle — fails
at runtime rather than at compile time. Debugging requires reading the
`routing_log` field we append to state at every hop, since there is no structural
diagram to fall back on. Visualisation must be reconstructed from execution traces
rather than graph metadata. For production systems, this demands rigorous node-level
unit tests that assert not just output values but also the `goto` destination
returned under each branch condition.

---

## 2. The Reality of Black-Box XAI in Language Models

LIME and SHAP were designed for models with stable, differentiable output
distributions — classifiers that return a probability vector you can measure
before and after perturbation. LLMs do not expose raw token probabilities through
standard API calls, and their outputs are non-deterministic: the same masked input
may produce a different response on two consecutive invocations due to temperature
sampling.

This forces proxy implementations. Our proxy LIME measures cosine similarity
between TF-IDF vectors of original and masked log text rather than a true
confidence drop. Our proxy SHAP computes Shapley values over an anomaly score
derived from z-score distance rather than a model decision boundary. Both
approximations are post-hoc and indirect — they explain the log data's internal
structure, not the LLM's generative process.

The reliability of these estimates is bounded by how well TF-IDF similarity
proxies semantic confidence. For structured error logs with consistent vocabulary,
the proxy is reasonable. For free-form reasoning traces where token order and
context carry most of the meaning, masking tokens independently violates the
independence assumption that both LIME and SHAP rely on. Results should be
treated as directional signals — "latency_ms was the strongest numeric driver of
anomalous behaviour in this session" — rather than precise causal claims.

---

## 3. Static Fallbacks vs. Dynamic Context Self-Healing

`RunnableWithFallbacks` is deterministic and cheap. The fallback chain is defined
at construction time; when the primary runnable fails, LangChain routes to the
next entry in the list with zero additional LLM calls until `_SelfHealRunnable`
fires. Latency overhead is bounded and predictable. The cost is inflexibility —
a static fallback cannot adapt its recovery strategy based on the nature of the
failure.

Dynamic LLM self-healing (`_SelfHealRunnable`) injects the caught error trace
into a corrective prompt and asks the model to reason about recovery. This can
handle novel failure modes that no static handler anticipated, producing responses
that are contextually appropriate rather than generic. The cost is significant:
each self-heal attempt consumes one full LLM call at normal token cost, adds
unpredictable latency, and introduces a new failure surface — the self-heal
itself can fail if the model misreads the error context.

In a production multi-agent system the correct architecture layers both. Use
`RunnableWithRetry` for transient infrastructure faults that are statistically
likely to resolve on retry (network blips, rate limits). Use static
`RunnableWithFallbacks` for known application-level error types where the
recovery action is well-defined and cheap. Reserve LLM self-healing for the
residual category of unexpected semantic failures where no static handler
applies — and always append a hardcoded absolute fallback as the final safety
net to guarantee the system never crashes unhandled regardless of what the
self-healing LLM does.