"""
Phase 3 - RakFort IT Help Desk MCP server.
Exposes IT help desk tools over the Model Context Protocol (stdio transport).

Data is SIMULATED except get_system_info (real, read-only) and tickets (saved locally).
IMPORTANT: never use print() in a stdio MCP server - stdout carries the protocol.
"""
import json
import os
import platform
import re
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RakFort IT Help Desk", host="127.0.0.1", port=int(os.getenv("MCP_PORT", "8003")))

# ---------- Simulated data ----------
SERVICE_STATUS = {
    "email": "operational",
    "vpn": "degraded - some users report slow connections; engineers are investigating",
    "wifi": "operational",
    "printing": "operational",
    "company portal": "operational",
    "shared drives": "outage - unavailable since 09:15; fix in progress (incident INC-1042)",
}

CONNECTIVITY = {
    "internet": "reachable - 18 ms",
    "vpn.rakfort.example": "reachable - 42 ms (slower than normal - see VPN status)",
    "mail.rakfort.example": "reachable - 25 ms",
    "intranet.rakfort.example": "reachable - 22 ms",
    "files.rakfort.example": "unreachable - request timed out (shared drives outage INC-1042)",
}

# ---------- Ticket storage and validation rules ----------
DATA_DIR = Path(__file__).parent / "data"
TICKETS_FILE = DATA_DIR / "tickets.json"

CATEGORIES = {"network", "account", "email", "printing", "software", "hardware", "other"}
PRIORITIES = {"P1", "P2", "P3", "P4"}
TICKET_ID_PATTERN = re.compile(r"^TKT-\d{4}$")
SECRET_PATTERN = re.compile(r"\b(password|passwd|pwd)\b\s*(is|=|:)", re.IGNORECASE)


def _load_tickets() -> list[dict]:
    """Private helper (not a tool): read all tickets from disk."""
    if not TICKETS_FILE.exists():
        return []
    return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))


def _save_tickets(tickets: list[dict]) -> None:
    """Private helper (not a tool): write all tickets to disk."""
    DATA_DIR.mkdir(exist_ok=True)
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


# ---------- Tools ----------
@mcp.tool()
def check_service_status(service: str) -> str:
    """Check the current status of a RakFort IT service.

    Use this when a user reports that a service seems down, slow or not working,
    BEFORE giving troubleshooting steps - there may be a known outage.
    Valid services: email, vpn, wifi, printing, company portal, shared drives.
    """
    key = service.strip().lower()
    if key not in SERVICE_STATUS:
        return f"Unknown service '{service}'. Known services: {', '.join(SERVICE_STATUS)}."
    return f"{key}: {SERVICE_STATUS[key]}"


@mcp.tool()
def create_ticket(summary: str, description: str, category: str, priority: str = "P3") -> str:
    """Create an IT support ticket and return its ticket ID.

    Only call this AFTER the user has clearly confirmed they want a ticket raised.
    category: one of network, account, email, printing, software, hardware, other.
    priority: P1 = whole office or critical system down, P2 = a team affected,
              P3 = one person affected with a workaround (default), P4 = request or question.
    Never include passwords or other secrets in the summary or description.
    """
    summary, description = summary.strip(), description.strip()
    category, priority = category.strip().lower(), priority.strip().upper()

    if not 5 <= len(summary) <= 120:
        return "Error: summary must be between 5 and 120 characters."
    if not 10 <= len(description) <= 2000:
        return "Error: description must be between 10 and 2000 characters."
    if category not in CATEGORIES:
        return f"Error: category must be one of: {', '.join(sorted(CATEGORIES))}."
    if priority not in PRIORITIES:
        return "Error: priority must be P1, P2, P3 or P4."
    if SECRET_PATTERN.search(f"{summary} {description}"):
        return ("Error: the ticket text appears to contain a password. "
                "Remove it and try again - passwords must never be put in tickets.")

    tickets = _load_tickets()
    ticket_id = f"TKT-{1001 + len(tickets)}"
    tickets.append({
        "id": ticket_id,
        "summary": summary,
        "description": description,
        "category": category,
        "priority": priority,
        "status": "open",
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    _save_tickets(tickets)
    return f"Ticket {ticket_id} created ({priority}, {category}): {summary}"


@mcp.tool()
def get_ticket_status(ticket_id: str) -> str:
    """Look up the status of an existing IT support ticket by its ID (format: TKT-1234)."""
    ticket_id = ticket_id.strip().upper()
    if not TICKET_ID_PATTERN.match(ticket_id):
        return "Error: ticket IDs look like TKT-1001."

    for ticket in _load_tickets():
        if ticket["id"] == ticket_id:
            return (f"{ticket['id']}: {ticket['status']} | {ticket['priority']} | "
                    f"{ticket['category']} | {ticket['summary']} (created {ticket['created']})")
    return f"No ticket found with ID {ticket_id}."


@mcp.tool()
def check_connectivity(target: str) -> str:
    """Test whether a RakFort system is reachable on the network (simulated).

    Use this to tell a local problem apart from a server-side outage.
    Valid targets: internet, vpn.rakfort.example, mail.rakfort.example,
    intranet.rakfort.example, files.rakfort.example.
    """
    key = target.strip().lower()
    if key not in CONNECTIVITY:   # allow-list: arbitrary hosts/commands are never accepted
        return f"Unknown target '{target}'. Valid targets: {', '.join(CONNECTIVITY)}."
    return f"{key}: {CONNECTIVITY[key]}"


@mcp.tool()
def get_system_info() -> str:
    """Get basic, non-sensitive details of the computer running the help desk service:
    operating system, version, architecture, CPU cores and Python version.
    Use when troubleshooting needs system details.
    """
    # Data minimisation: no hostname, username or IP address is returned
    return (
        f"Operating system: {platform.system()} {platform.release()} (build {platform.version()})\n"
        f"Architecture: {platform.machine()}\n"
        f"CPU cores: {os.cpu_count()}\n"
        f"Python: {platform.python_version()}"
    )


# ---------- Resources (read-only data, fetched by the application) ----------
@mcp.resource("helpdesk://catalogue")
def service_catalogue() -> str:
    """Catalogue of what this help desk server knows about: monitored services,
    connectivity targets, ticket categories and ticket priorities."""
    # Built from the same data the tools use - a single source of truth
    return json.dumps(
        {
            "monitored_services": list(SERVICE_STATUS),
            "connectivity_targets": list(CONNECTIVITY),
            "ticket_categories": sorted(CATEGORIES),
            "ticket_priorities": sorted(PRIORITIES),
        },
        indent=2,
    )

if __name__ == "__main__":
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "sse"))

