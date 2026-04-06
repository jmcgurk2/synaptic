import os
from collections import defaultdict
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlmodel import Session, select

from database import Entry, get_engine
from mattermost import post_message

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def build_digest() -> str:
    """Build the morning digest markdown."""
    engine = get_engine()
    now = datetime.utcnow()
    sections = []

    with Session(engine) as session:
        # Entries tagged #pending or #review older than 48h
        cutoff_48h = now - timedelta(hours=48)
        stale = session.exec(
            select(Entry).where(
                Entry.created_at < cutoff_48h,
                Entry.tags.contains("pending") | Entry.tags.contains("review"),
            )
        ).all()
        if stale:
            lines = []
            for e in stale[:20]:
                proj_str = f" [{e.project}]" if e.project else ""
                lines.append(f"- **{e.title}** ({e.type}){proj_str} — {e.summary}")
            sections.append(
                "### Stale items (>48h, tagged #pending or #review)\n" + "\n".join(lines)
            )

        # Projects with updated_at older than 7 days
        cutoff_7d = now - timedelta(days=7)
        dormant = session.exec(
            select(Entry).where(
                Entry.type == "Project",
                Entry.updated_at < cutoff_7d,
            )
        ).all()
        if dormant:
            lines = []
            for e in dormant[:20]:
                proj_str = f" [{e.project}]" if e.project else ""
                lines.append(f"- **{e.title}**{proj_str} — last updated {e.updated_at:%Y-%m-%d}")
            sections.append(
                "### Dormant projects (no update in 7+ days)\n" + "\n".join(lines)
            )

        # 5 most recent captures
        recent = session.exec(
            select(Entry).order_by(Entry.created_at.desc()).limit(5)
        ).all()
        if recent:
            lines = [f"- **{e.title}** ({e.type}, {e.source}) — {e.summary}" for e in recent]
            sections.append("### Recent captures grouped by project\n" + "\n".join(lines))
        recent = session.exec(
            select(Entry).order_by(Entry.created_at.desc()).limit(15)
        ).all()
        if recent:
            # Group by project
            by_project = defaultdict(list)
            for e in recent:
                key = e.project or "ungrouped"
                by_project[key].append(e)
            
            recent_lines = []
            for proj, ents in by_project.items():
                if proj == "ungrouped":
                    for e in ents:
                        recent_lines.append(f"- **{e.title}** ({e.type}, {e.source}) — {e.summary}")
                else:
                    recent_lines.append(f"**[{proj}]**")
                    for e in ents:
                        recent_lines.append(f"  - **{e.title}** ({e.type}) — {e.summary}")
            sections.append(f"### Recent captures\n" + "\n".join(recent_lines))

    if not sections:
        return "**Synaptic Morning Digest**\n\nNothing to report — your second brain is quiet."

    header = f"**Synaptic Morning Digest** — {now:%A, %B %d}\n"
    return header + "\n\n".join(sections)


async def send_digest():
    """Build digest and post to Mattermost."""
    channel_id = os.getenv("MATTERMOST_DIGEST_CHANNEL_ID", "")
    if not channel_id:
        logger.warning("MATTERMOST_DIGEST_CHANNEL_ID not set — skipping digest")
        return

    digest = await build_digest()
    await post_message(channel_id, digest)
    logger.info("Digest posted to channel %s", channel_id)


def init_scheduler():
    """Start APScheduler with the digest cron job."""
    global _scheduler

    cron_expr = os.getenv("DIGEST_CRON", "0 8 * * *")
    parts = cron_expr.split()
    if len(parts) != 5:
        logger.error("Invalid DIGEST_CRON: %s — falling back to 0 8 * * *", cron_expr)
        parts = ["0", "8", "*", "*", "*"]

    trigger = CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
    )

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(send_digest, trigger, id="morning_digest")
    _scheduler.start()
    logger.info("Scheduler started — digest cron: %s", cron_expr)


def shutdown_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
