# Synaptic Roadmap

Personal AI knowledge platform — capture, classify, connect, recall.

## Completed

- **Capture & classify** — Mattermost webhook ingestion, LiteLLM-powered classification (type, title, tags, summary, confidence)
- **Dual-store search** — SQLite text search + Qdrant semantic vector search, merged and deduplicated
- **Project-aware operations** — `project` field on entries, project-filtered search/reports/digest, `!projects` command
- **Recall engine** — `/recall` REST endpoint + `!recall` / `!brief` Mattermost commands; LiteLLM synthesises narrative answers from stored knowledge
- **Morning digest** — Scheduled daily briefing with recent captures grouped by project, stale/dormant item alerts
- **Capture mode** — Intent detection for commands (`!search`, `!report`, `!recent`, `!recall`, `!brief`, `!projects`, `!fix`) vs. plain capture

---

## Phase 1 — Smarter Capture

### Auto-linking
When a new entry is captured, run semantic search against existing entries. If similarity score exceeds a threshold, store bidirectional links (new `entry_links` table). Surface these in recall and search results so the second brain builds its own connection graph over time.

### Richer extraction
Expand the classifier prompt to also extract:
- **Action items** — discrete tasks embedded in a note ("need to order X", "should follow up with Y")
- **Deadlines / dates** — parse relative and absolute dates, store as structured fields
- **People mentioned** — extract names, link to Contact-type entries
- **Auto-project assignment** — infer project from content + existing project entries, populate the `project` field automatically instead of requiring manual tags

### Multi-turn capture (`!expand`)
New command that takes an entry ID and starts a conversation: LiteLLM reads the entry, asks follow-up questions via Mattermost ("What's the timeline?", "Who else is involved?", "How does this relate to X?"), and enriches the entry with each response. Turns sparse notes into rich knowledge.

---

## Phase 2 — Proactive Resurfacing

### Capture-time connections
When a new entry is stored, automatically surface related entries back to the user: "This connects to 3 entries about Orex from last month." Runs as part of the webhook response, not a separate command.

### Spaced resurfacing
Periodically resurface important entries the user hasn't interacted with recently. Not random — weighted by: entry type (Projects > Ideas > Admin), confidence score, number of connections, and recency of last recall. Integrated into the morning digest.

### Weekly synthesis
A scheduled weekly job (in addition to the daily digest) that:
- Identifies emerging themes across the week's captures
- Flags projects with activity vs. projects gone quiet
- Connects dots the user might have missed ("You captured 4 entries about network security this week — is this becoming a project?")
- Summarises open action items across all projects

---

## Phase 3 — Source Integrations

### Email capture
Forward-to-capture email address (or IMAP polling) that ingests emails as entries. Classifier extracts the relevant knowledge from the email body, strips signatures/headers, and stores the distilled content.

### Web clipper
REST endpoint (`/capture/url`) that accepts a URL, fetches the content, extracts the meaningful text, and runs it through the classifier. Could be called from a bookmarklet, browser extension, or n8n workflow.

### Voice capture (Orex)
Integration point for the Orex voice assistant — spoken notes transcribed and sent to Synaptic's capture endpoint. Orex becomes the voice layer; Synaptic remains the knowledge store.

### API / webhook sources
Generic webhook endpoint that accepts structured payloads from other tools (GitHub events, monitoring alerts, calendar events). Each source gets a classifier hint to improve categorisation.

---

## Phase 4 — Knowledge Graph

### Connection graph
Build an explicit graph layer on top of the entry store:
- Nodes = entries
- Edges = auto-detected semantic links (from Phase 1) + manual links
- Store in Qdrant metadata or a lightweight graph table in SQLite

### Graph-aware recall
When recalling, traverse the graph to pull in connected entries even if they don't directly match the query. "Tell me about Orex" should also surface the hardware platform entry, the voice capture integration plan, and any related infrastructure notes.

### Visualisation
Optional web UI (or Mattermost-rendered) that shows the knowledge graph — clusters of related entries, orphaned notes that need connections, project clusters. Not a priority but a natural extension once the graph data exists.

---

## Infrastructure & Pipeline

### Build optimisation
- BuildKit permanently disabled on VM 120 (done — daemon.json + /etc/environment)
- Docker layer cache must be preserved — never prune build cache
- Consider pre-building a base image with torch/embeddings to speed up code-only rebuilds

### Deployment pipeline
All deploys follow the MCP pipeline:
1. Write code → SSH MCP to VM 120
2. Push to GitHub → GitHub MCP on VM 110 (`mcp-call github push_files`)
3. Deploy → SSH MCP to VM 120 (`DOCKER_BUILDKIT=0 docker compose up -d --build`)

No manual git. No ad-hoc Docker troubleshooting.
