from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("analysis_dashboard.graph_client")


class Neo4jGraphClient:
    """
    Thread-safe Neo4j Aura DB client for writing HLP interaction graphs.

    Parameters
    ----------
    uri      : neo4j+s://... Aura DB connection URI
    username : typically "neo4j"
    password : Aura DB password
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        self.uri      = uri
        self.username = username
        self.password = password
        self._driver  = None
        self._connected = False

    def connect(self) -> bool:
        """Open the Neo4j driver. Returns True on success."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("Connected to Neo4j Aura DB at %s", self.uri)
            self._ensure_constraints()
            return True
        except Exception as exc:
            logger.error("Neo4j connection failed: %s", exc)
            self._connected = False
            return False

    def _ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't exist."""
        constraints = [
            "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
            "CREATE CONSTRAINT agent_action_id_unique IF NOT EXISTS FOR (a:AgentAction) REQUIRE a.action_id IS UNIQUE",
            "CREATE CONSTRAINT mcp_call_id_unique IF NOT EXISTS FOR (m:MCPServerCall) REQUIRE m.call_id IS UNIQUE",
        ]
        with self._driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as exc:
                    logger.debug("Constraint may already exist: %s", exc)
        logger.info("Neo4j schema constraints verified.")

    # ── Write operations ────────────────────────────────────────

    def upsert_session(self, session_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        MERGE a (:Session) node.
        Returns a summary of what was written.
        """
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        query = """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET
            s.created_at      = $created_at,
            s.primary_model   = $primary_model,
            s.sampling_model  = $sampling_model,
            s.query_count     = $query_count,
            s.total_entries   = $total_entries
        ON MATCH SET
            s.query_count     = $query_count,
            s.total_entries   = $total_entries
        RETURN s.session_id as session_id, labels(s) as labels
        """
        with self._driver.session() as neo_session:
            result = neo_session.run(query, session_id=session_id, **metadata)
            record = result.single()
        summary = {"session_id": session_id, "node": "Session", "operation": "MERGE"}
        logger.info("Upserted Session node: %s", session_id[:8])
        return summary

    def upsert_agent_action(
        self,
        action_id: str,
        session_id: str,
        action_type: str,
        content_preview: str,
        timestamp: str,
        latency_ms: float | None = None,
        namespace: str = "",
    ) -> dict[str, Any]:
        """
        MERGE an (:AgentAction) node and create [:TRIGGERED] edge from its session.
        """
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        query = """
        MERGE (a:AgentAction {action_id: $action_id})
        ON CREATE SET
            a.session_id      = $session_id,
            a.action_type     = $action_type,
            a.content_preview = $content_preview,
            a.timestamp       = $timestamp,
            a.latency_ms      = $latency_ms,
            a.namespace       = $namespace
        ON MATCH SET
            a.latency_ms      = $latency_ms

        WITH a
        MATCH (s:Session {session_id: $session_id})
        MERGE (s)-[:TRIGGERED]->(a)
        RETURN a.action_id as action_id
        """
        with self._driver.session() as neo_session:
            neo_session.run(
                query,
                action_id=action_id,
                session_id=session_id,
                action_type=action_type,
                content_preview=content_preview[:200],
                timestamp=timestamp,
                latency_ms=latency_ms,
                namespace=namespace,
            )
        return {
            "action_id":  action_id,
            "node":       "AgentAction",
            "edge":       "TRIGGERED",
            "operation":  "MERGE",
        }

    def upsert_mcp_server_call(
        self,
        call_id: str,
        session_id: str,
        tool_name: str,
        interaction_type: str,
        content_preview: str,
        timestamp: str,
        latency_ms: float | None = None,
        namespace: str = "",
    ) -> dict[str, Any]:
        """
        MERGE an (:MCPServerCall) node.
        """
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        query = """
        MERGE (m:MCPServerCall {call_id: $call_id})
        ON CREATE SET
            m.session_id      = $session_id,
            m.tool_name       = $tool_name,
            m.interaction_type = $interaction_type,
            m.content_preview = $content_preview,
            m.timestamp       = $timestamp,
            m.latency_ms      = $latency_ms,
            m.namespace       = $namespace
        ON MATCH SET
            m.latency_ms      = $latency_ms
        RETURN m.call_id as call_id
        """
        with self._driver.session() as neo_session:
            neo_session.run(
                query,
                call_id=call_id,
                session_id=session_id,
                tool_name=tool_name,
                interaction_type=interaction_type,
                content_preview=content_preview[:200],
                timestamp=timestamp,
                latency_ms=latency_ms,
                namespace=namespace,
            )
        return {
            "call_id":   call_id,
            "node":      "MCPServerCall",
            "operation": "MERGE",
        }

    def link_action_to_call(
        self,
        action_id: str,
        call_id: str,
        edge_type: str = "ROUTED_TO",
    ) -> dict[str, Any]:
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        if edge_type not in ("ROUTED_TO", "DEPENDS_ON"):
            edge_type = "ROUTED_TO"

        query = f"""
        MATCH (a:AgentAction   {{action_id: $action_id}})
        MATCH (m:MCPServerCall {{call_id:   $call_id}})
        MERGE (a)-[r:{edge_type}]->(m)
        RETURN type(r) as edge_type
        """
        with self._driver.session() as neo_session:
            result = neo_session.run(query, action_id=action_id, call_id=call_id)
            record = result.single()
        return {
            "from":      action_id,
            "to":        call_id,
            "edge":      edge_type,
            "operation": "MERGE",
        }

    def link_call_to_call(
        self,
        from_call_id: str,
        to_call_id: str,
        edge_type: str = "DEPENDS_ON",
    ) -> dict[str, Any]:
        """Link two MCPServerCall nodes (e.g. sampling chain)."""
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        query = f"""
        MATCH (a:MCPServerCall {{call_id: $from_call_id}})
        MATCH (b:MCPServerCall {{call_id: $to_call_id}})
        MERGE (a)-[r:{edge_type}]->(b)
        RETURN type(r) as edge_type
        """
        with self._driver.session() as neo_session:
            neo_session.run(query, from_call_id=from_call_id, to_call_id=to_call_id)
        return {
            "from":      from_call_id,
            "to":        to_call_id,
            "edge":      edge_type,
            "operation": "MERGE",
        }

    # ── Read operations ─────────────────────────────────────────

    def get_session_graph(self, session_id: str) -> dict[str, Any]:
        """Fetch all nodes and edges for a session."""
        if not self._connected:
            return {"error": "Not connected to Neo4j"}

        node_query = """
        MATCH (n) WHERE n.session_id = $session_id
        RETURN labels(n) as labels, properties(n) as props
        """
        edge_query = """
        MATCH (a)-[r]->(b)
        WHERE a.session_id = $session_id OR b.session_id = $session_id
        RETURN type(r) as edge_type, properties(a) as from_props, properties(b) as to_props
        """
        with self._driver.session() as neo_session:
            nodes = [
                {"labels": r["labels"], "props": dict(r["props"])}
                for r in neo_session.run(node_query, session_id=session_id)
            ]
            edges = [
                {
                    "type":      r["edge_type"],
                    "from":      dict(r["from_props"]),
                    "to":        dict(r["to_props"]),
                }
                for r in neo_session.run(edge_query, session_id=session_id)
            ]
        return {"nodes": nodes, "edges": edges}

    def get_graph_summary(self) -> dict[str, Any]:
        """High-level graph stats for the dashboard."""
        if not self._connected:
            return {
                "sessions": 0, "agent_actions": 0,
                "mcp_calls": 0, "edges": 0, "connected": False,
            }
        with self._driver.session() as neo_session:
            sessions      = neo_session.run("MATCH (s:Session) RETURN count(s) as c").single()["c"]
            agent_actions = neo_session.run("MATCH (a:AgentAction) RETURN count(a) as c").single()["c"]
            mcp_calls     = neo_session.run("MATCH (m:MCPServerCall) RETURN count(m) as c").single()["c"]
            edges         = neo_session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        return {
            "sessions":      sessions,
            "agent_actions": agent_actions,
            "mcp_calls":     mcp_calls,
            "edges":         edges,
            "connected":     True,
        }

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._connected = False


# ─────────────────────────────────────────────────────────────
# Graph projection logic
# Maps log entries from HLPLogStore → Neo4j nodes/edges
# ─────────────────────────────────────────────────────────────

def project_logs_to_graph(
    log_entries: list[dict],
    graph: Neo4jGraphClient,
) -> dict[str, Any]:
    """
    Extract relationships from log entry metadata and project them into Neo4j.

    Mapping rules:
      • Every unique session_id → (:Session) node
      • agent_reasoning / agent_final_answer entries → (:AgentAction) nodes
        with [:TRIGGERED] from their (:Session)
      • tool_invocation / resource_read entries → (:MCPServerCall) nodes
      • sampling_request entries → (:MCPServerCall) with interaction_type='sampling'
      • tool_invocation AgentAction → [:ROUTED_TO] → MCPServerCall for same tool+session
      • sampling_request → [:DEPENDS_ON] → the tool call that triggered it

    Returns a commit summary dict shown in the Streamlit dashboard.
    """
    commits: list[dict] = []
    errors:  list[str]  = []

    # ── 1. Group entries by session ──────────────────────────────
    by_session: dict[str, list[dict]] = {}
    for entry in log_entries:
        sid = entry.get("session_id", "unknown")
        by_session.setdefault(sid, []).append(entry)

    for session_id, entries in by_session.items():
        # ── 2. Upsert (:Session) node ────────────────────────────
        try:
            result = graph.upsert_session(
                session_id=session_id,
                metadata={
                    "created_at":     entries[0].get("timestamp", ""),
                    "primary_model":  "unknown",
                    "sampling_model": "unknown",
                    "query_count":    sum(
                        1 for e in entries if e.get("interaction_type") == "agent_final_answer"
                    ),
                    "total_entries":  len(entries),
                },
            )
            commits.append(result)
        except Exception as exc:
            errors.append(f"Session upsert failed: {exc}")

        # ── 3. Agent actions ─────────────────────────────────────
        agent_entries = [
            e for e in entries
            if e.get("interaction_type") in ("agent_reasoning", "agent_final_answer")
        ]
        for entry in agent_entries:
            action_id = f"action_{entry.get('store_key', entry.get('id', ''))}"
            try:
                result = graph.upsert_agent_action(
                    action_id=action_id,
                    session_id=session_id,
                    action_type=entry.get("interaction_type", "agent_reasoning"),
                    content_preview=entry.get("content", "")[:200],
                    timestamp=entry.get("timestamp", ""),
                    latency_ms=entry.get("latency_ms"),
                    namespace=entry.get("namespace", ""),
                )
                commits.append(result)
            except Exception as exc:
                errors.append(f"AgentAction upsert failed: {exc}")

        # ── 4. MCP Server calls ──────────────────────────────────
        server_entries = [
            e for e in entries
            if e.get("interaction_type") in (
                "tool_invocation", "resource_read",
                "sampling_request", "sampling_response",
            )
        ]
        prev_tool_call_id: dict[str, str] = {}  # tool_name → last call_id

        for entry in server_entries:
            call_id = f"call_{entry.get('store_key', entry.get('id', ''))}"
            tool_name = entry.get("tool_name") or "unknown"
            try:
                result = graph.upsert_mcp_server_call(
                    call_id=call_id,
                    session_id=session_id,
                    tool_name=tool_name,
                    interaction_type=entry.get("interaction_type", "tool_invocation"),
                    content_preview=entry.get("content", "")[:200],
                    timestamp=entry.get("timestamp", ""),
                    latency_ms=entry.get("latency_ms"),
                    namespace=entry.get("namespace", ""),
                )
                commits.append(result)

                # Link sampling_request → DEPENDS_ON the preceding tool call
                itype = entry.get("interaction_type", "")
                if itype == "sampling_request" and prev_tool_call_id:
                    last_tool = list(prev_tool_call_id.values())[-1]
                    edge_result = graph.link_call_to_call(
                        from_call_id=last_tool,
                        to_call_id=call_id,
                        edge_type="DEPENDS_ON",
                    )
                    commits.append(edge_result)

                if itype in ("tool_invocation", "resource_read"):
                    prev_tool_call_id[tool_name] = call_id

            except Exception as exc:
                errors.append(f"MCPServerCall upsert failed: {exc}")

        # ── 5. Link AgentAction → MCPServerCall ──────────────────
        # Match by session + temporal ordering (agent call precedes server call)
        for agent_entry in agent_entries:
            action_id = f"action_{agent_entry.get('store_key', agent_entry.get('id', ''))}"
            a_ts = agent_entry.get("timestamp", "")
            # Find the first server call after this agent action
            for server_entry in server_entries:
                s_ts = server_entry.get("timestamp", "")
                if s_ts >= a_ts:
                    call_id = f"call_{server_entry.get('store_key', server_entry.get('id', ''))}"
                    try:
                        edge_result = graph.link_action_to_call(
                            action_id=action_id,
                            call_id=call_id,
                            edge_type="ROUTED_TO",
                        )
                        commits.append(edge_result)
                    except Exception as exc:
                        errors.append(f"ROUTED_TO edge failed: {exc}")
                    break  # only link to the immediate next call

    return {
        "commits": commits,
        "errors":  errors,
        "sessions_processed": len(by_session),
        "total_commits":      len(commits),
    }