# Task: Deploy Synaptic to MOHAWK

## Prerequisites
- Proxmox VM 120 provisioned (Ubuntu 24.04, 2 vCPU, 4GB RAM, 60GB disk, VLAN 55, IP 10.55.55.20)
- Repo cloned or copied to /opt/synaptic on the VM
- .env file populated from .env.example

## Steps

### 1. Install Docker on VM
SSH to 10.55.55.20 and run the Docker install script for Ubuntu 24.04.

### 2. Start the stack
```bash
cd /opt/synaptic
docker compose up -d
docker compose ps
```

### 3. Verify health
```bash
curl http://10.55.55.20:8000/health
```
Expect: `{"status":"ok","sqlite":"ok","qdrant":"ok"}`

### 4. Register Zoraxy proxy + DNS + Cloudflare tunnel
Via Ansible on VM 112:
```bash
docker exec ansible-runner ansible-playbook /ansible/playbooks/zoraxy-proxy-manage.yml \
  -e action=add \
  -e hostname=synaptic.mohawkops.ai \
  -e target=10.55.55.20:8000 \
  -e create_dns=true
```

### 5. Add Uptime Kuma monitors
Via kuma-mcp (VM 110:8011):
- synaptic-api → https://synaptic.mohawkops.ai/health
- synaptic-qdrant → http://10.55.55.20:6333/healthz

### 6. Register in NetBox
```bash
curl -s -X POST http://10.44.44.10:8000/api/ipam/services/ \
  -H "Authorization: Token <NETBOX_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"virtual_machine": 120, "name": "synaptic-api", "ports": [8000], "protocol": "tcp"}'

curl -s -X POST http://10.44.44.10:8000/api/ipam/services/ \
  -H "Authorization: Token <NETBOX_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"virtual_machine": 120, "name": "synaptic-qdrant", "ports": [6333], "protocol": "tcp"}'
```

### 7. Install Wazuh agent
```bash
curl -so wazuh-agent.deb https://packages.wazuh.com/4.x/apt/pool/main/w/wazuh-agent/wazuh-agent_4.7.0-1_amd64.deb
sudo WAZUH_MANAGER='10.44.44.10' dpkg -i ./wazuh-agent.deb
sudo systemctl enable wazuh-agent && sudo systemctl start wazuh-agent
```

### 8. Configure Mattermost bot
In Mattermost System Console:
1. Integrations → Bot Accounts → Add Bot (username: synaptic)
2. Copy token → MATTERMOST_BOT_TOKEN in .env
3. Create channel #synaptic-brain, copy channel ID → MATTERMOST_DIGEST_CHANNEL_ID in .env
4. Integrations → Outgoing Webhooks → Add
   - Channel: #synaptic-brain
   - Callback URL: https://synaptic.mohawkops.ai/webhook
   - Copy token → MATTERMOST_WEBHOOK_TOKEN in .env
5. Restart the API container: `docker compose restart api`

### 9. Verify end to end
Send a test message to @synaptic in #synaptic-brain:
"Test capture — new project idea for Synaptic deployment"
Expect a confirmation reply from the bot.
