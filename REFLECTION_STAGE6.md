# Reflection — Stage 6

## Cosine Metric Spaces vs. Euclidean Distances

Our Stage 5 CRAG index looked correct on paper — a RedisVL schema declared
with `distance_metric: "cosine"` — but it was never actually queried. Every
similarity score came from a hand-rolled `numpy.dot`/`numpy.linalg.norm`
loop over every knowledge-base item in Python. Functionally that loop *was*
computing cosine similarity, so the choice of metric was already the right
one; the Stage 6 fix wasn't "switch to cosine," it was "actually let the
vector index do this via a real ANN query" instead of an unindexed,
linearly-scaled Python scan.

That said, the choice of cosine over raw L2/Euclidean distance still
matters, and matters more as the system scales:

- **Magnitude invariance.** Our knowledge base has domain summaries (a
  couple of sentences), section summaries, and leaf chunks (full paragraphs
  with code). Embedding magnitude tends to grow with input length for most
  sentence-embedding models, even when the model isn't supposed to encode
  length as meaning. Euclidean distance conflates "this document is long"
  with "this document is different" — a short, precisely on-topic query
  embedding can end up numerically far from a long, on-topic chunk purely
  because of vector magnitude, not semantic content. Cosine distance strips
  magnitude out entirely and measures only the angle between vectors, so a
  three-word query and a 300-word chunk about the same concept can still
  score as highly similar.
- **Stability in high dimensions.** As embedding dimensionality grows
  (768 in our case), the well-known "curse of dimensionality" pushes
  Euclidean distances between random points closer together in relative
  terms — the gap between "similar" and "dissimilar" pairs shrinks as a
  proportion of the average distance, which erodes ranking quality.
  Cosine similarity, by normalizing to the unit hypersphere before
  comparing, is far less sensitive to this compression effect, because it's
  comparing *directions* rather than raw point-to-point distances in a
  space where "far" stops meaning much.
- **What this actually bought us structurally.** By moving to real
  `VectorQuery` calls against a RedisVL index (and an equivalent `<=>`
  cosine-distance query against Supabase/pgvector), we get sub-linear ANN
  search instead of an O(n) Python loop, and the distance metric is now a
  first-class property of the index rather than an implementation detail
  buried in application code — so a future contributor can't silently swap
  in an inconsistent metric (say, average a Euclidean score with a cosine
  score) without the index schema itself changing.

## The Vulnerabilities of Cross-Layer Attack Surfaces in x402 Flows

x402 sits at an awkward seam: the HTTP/MCP layer above it is synchronous
and expects an answer in milliseconds, while the blockchain layer below it
is asynchronous, probabilistic, and can take seconds to minutes to reach
finality. A few concrete failure modes fall directly out of that mismatch,
and the mitigations we built in (or left as follow-up work):

- **Signature replay.** An EIP-3009 `transferWithAuthorization` is a
  bearer-style signed message — if a byte-for-byte payload gets replayed,
  naive verification could try to double-spend the same authorization.
  EIP-3009 itself blocks the most naive form of this by requiring a fresh,
  random 32-byte `nonce` per authorization (see `wallet.py:sign_transfer_authorization`,
  which generates one via `secrets.token_hex(32)` on every call rather than
  an incrementing counter) and by having the token contract track "used"
  nonces on-chain — once a nonce is consumed, resubmitting the identical
  signed payload reverts. The remaining exposure is at the facilitator
  layer, not the contract layer: our `/verify` endpoint should be treated
  as advisory only, since a payload can pass `/verify` and then either
  never be settled, or be settled by a *different* concurrent request
  racing for the same nonce. Our paywall only trusts the `/settle`
  response, not `/verify`, as the basis for actually granting continued
  access — `verify` just avoids wasting an LLM call on an obviously-bad
  payload before we've done the expensive step.
- **The verify/settle race window.** Because `/settle` waits on
  non-deterministic block inclusion, there's a window where our server has
  already run the (paid) tool logic optimistically after `/verify` passed,
  but before `/settle` confirms the transfer actually landed. If settlement
  later fails (facilitator can't relay, nonce got front-run, chain
  reorganizes), the tool result was already returned. We call this out
  explicitly in `paywall.py` as a "billing gap, not a correctness bug" —
  the mitigation isn't to block on-chain finality synchronously (that would
  make every paid call take as long as a block confirmation, which defeats
  the point of a fast agent loop), it's to treat a failed settlement as an
  out-of-band reconciliation event: log it to the audit trail with
  `success: false`, and use that signal to gate *future* requests from the
  same payer rather than trying to claw back the one that already ran.
- **Chain reorganization / finality assumptions.** A transaction can appear
  "confirmed" at one block depth and then be reorganized out on a
  short-lived fork, especially on a testnet with fewer validators than
  mainnet. Treating "the facilitator returned a transaction hash" as
  equivalent to "the funds have moved" is optimistic in exactly the way the
  brief warns about. The safer posture — which we've structured the audit
  trail to support even though we don't enforce it yet — is to treat a
  settlement as *provisional* until it's several blocks deep, and to
  reconcile any provisional settlement that later disappears by revoking
  the corresponding grant of access rather than assuming irreversibility
  the instant a hash comes back.
- **Session/ledger boundary confusion.** Because the FinOps ledger and the
  x402 audit trail are both keyed by an application-level `session_id`
  rather than an on-chain identity, there's a cross-layer trust boundary
  here too: a malicious client could reuse another session's `session_id`
  to make its own spend look like it's landing in someone else's (already
  budgeted) bucket. We don't currently authenticate `session_id` — it's
  supplied by the caller. The correct long-term mitigation is to derive the
  session key from something cryptographically tied to the caller (e.g. a
  hash of the payer's wallet address plus a server-issued session token)
  rather than trusting a client-supplied string, so the governance layer
  can't be gamed by mislabeling whose budget a call counts against.
