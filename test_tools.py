"""
Step 3.4 - Test each MCP tool directly, including invalid and malicious inputs.
"""
from helpdesk_server import (
    check_connectivity,
    check_service_status,
    create_ticket,
    get_system_info,
    get_ticket_status,
)

print("1. Valid ticket:     ", create_ticket(
    "Laptop cannot join RakFort-Corp Wi-Fi",
    "Since this morning my laptop fails to connect to RakFort-Corp. Restarted twice.",
    "network", "p3"))
print("2. Password in text: ", create_ticket(
    "Account locked out", "My password is Summer2026! please reset it", "account"))
print("3. Bad category:     ", create_ticket(
    "Printer making noise", "The floor 2 printer is jammed and grinding", "furniture"))
print("4. Lookup ticket:    ", get_ticket_status("tkt-1001"))
print("5. Malformed ID:     ", get_ticket_status("1001; DROP TABLE tickets"))
print("6. Connectivity:     ", check_connectivity("files.rakfort.example"))
print("7. Injection attempt:", check_connectivity("google.com && whoami"))
print("8. Service status:   ", check_service_status("vpn"))
print("9. System info:\n" + get_system_info())