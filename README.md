# Synaptic

A personal AI knowledge platform — a self-hosted second brain. Synaptic captures raw notes, classifies them with an LLM, stores them in both structured (SQLite) and semantic (Qdrant) storage, and surfaces them via search, digest, and context endpoints. It implements the 8 building blocks from Nate B. Jones' Second Brain framework: Capture, Sorter, Form, Filing Cabinet, Receipt, Bouncer, Tap on Shoulder, and Fix Button.

## Architecture

```
Mattermost (chat.mohawkops.ai)
    │
@synaptic bot account  ← outgoing webhook
    │
Synaptic agent service ← intent parsing, routes to API
    │
Synaptic API           ← THIS REPO (FastAPI on :8000)
    │
Qdrant + SQLite        ← on same VM
    │
LiteLLM proxy          ← local AI gateway (separate VM)
```

All knowledge is source-tagged — every entry records which agent wrote it (`@synaptic`, `@orex`, future agents). The API is agent-agnostic.

## Quick Start

```bash
git clone <repo-url> && cd synaptic
cp .env.example .env
# Edit .env with your Mattermost tokens and LiteLLM endpoint
docker compose up -d
curl http://localhost:8000/health
```

## API Endpoints

### POST /webhook
Entry point for Mattermost outgoing webhooks. Routes messages by intent:
- `!fix <type>` — reclassify last bounced entry
- `!search <query>` or `?<query>` — search knowledge base
- Anything else — capture as new knowledge

### POST /capture
Capture a new knowledge entry. Called by webhook handler and directly by agents.

```bash
curl -X POST http://localhost:8000/capture \
  -H "Content-Type: application/json" \
  -d '{"text": "Build a Terraform module for Proxmox VM provisioning", "source": "@synaptic"}'
```

Response:
```json
{
  "id": "uuid",
  "status": "stored",
  "type": "Project",
  "title": "Terraform module for Proxmox",
  "tags": ["terraform", "proxmox", "infrastructure"],
  "summary": "Build a Terraform module to automate Proxmox VM provisioning",
  "confidence": 0.92
}
```

### GET /search
Search across both SQLite (text) and Qdrant (semantic).

```bash
curl "http://localhost:8000/search?q=terraform&type=Project&limit=5"
```

### GET /recall/{id}
Full entry by ID including raw text and all metadata.

```bash
curl http://localhost:8000/recall/<entry-id>
```

### PATCH /entries/{id}
Fix button — reclassify an entry with a type hint.

```bash
curl -X PATCH http://localhost:8000/entries/<entry-id> \
  -H "Content-Type: application/json" \
  -d '{"type": "Task"}'
```

### GET /context
Returns 10 most recent entries and all pending-fix entries. Designed for agent context injection.

```bash
curl http://localhost:8000/context
```

### POST /digest
Trigger the morning digest immediately.

```bash
curl -X POST http://localhost:8000/digest
```

### GET /health
Health check — verifies SQLite and Qdrant connectivity.

```bash
curl http://localhost:8000/health
```

### GET /metrics
Prometheus-compatible metrics for Alloy scraping.

## Mattermost Bot Setup

1. **Create bot account**: System Console → Integrations → Bot Accounts → Add Bot (username: `synaptic`)
2. **Copy bot token** → `MATTERMOST_BOT_TOKEN` in `.env`
3. **Create channel** `#synaptic-brain`, copy channel ID → `MATTERMOST_DIGEST_CHANNEL_ID`
4. **Create outgoing webhook**: Integrations → Outgoing Webhooks
   - Channel: `#synaptic-brain`
   - Callback URL: `https://synaptic.mohawkops.ai/webhook`
   - Copy token → `MATTERMOST_WEBHOOK_TOKEN` in `.env`
5. **Restart**: `docker compose restart api`

## Usage Examples

**Capture a note** — send any message to `@synaptic` in `#synaptic-brain`:
```
@synaptic Set up automated backups for all Proxmox VMs using PBS
```

**Search** — use `?` prefix or `!search`:
```
@synaptic ?proxmox backups
@synaptic !search terraform
```

**Fix a bounced entry** — when the classifier isn't sure:
```
@synaptic !fix Task
```

## Deployment

See [cowork/deploy-synaptic.md](cowork/deploy-synaptic.md) for the full Cowork deployment task covering Docker setup, proxy registration, monitoring, and Mattermost configuration.

## Adding a New Agent Source

The API is agent-agnostic. Any service can write to Synaptic by calling `POST /capture` with a unique `source` tag:

```bash
curl -X POST http://synaptic.mohawkops.ai:8000/capture \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Discovered CVE-2024-1234 affects our nginx version",
    "source": "@orex"
  }'
```

The `source` field is stored on every entry and receipt, enabling filtering and attribution. No registration required — just pick a unique `@name` and start posting.
