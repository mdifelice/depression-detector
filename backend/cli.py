#!/usr/bin/env python3
"""CLI tool to manage the whitelisted emails for Depression Detector."""

import argparse
import json
import sys
from pathlib import Path

WHITELIST_FILE = Path(__file__).resolve().parent.parent / "data" / "whitelist.json"


def load_whitelist() -> list[str]:
    if not WHITELIST_FILE.exists():
        return []
    with open(WHITELIST_FILE, "r") as f:
        return json.load(f)


def save_whitelist(emails: list[str]):
    WHITELIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WHITELIST_FILE, "w") as f:
        json.dump(sorted(set(e.lower() for e in emails)), f, indent=2)


def cmd_add(args):
    emails = load_whitelist()
    email = args.email.lower()
    if email in emails:
        print(f"'{email}' is already whitelisted.")
        return
    emails.append(email)
    save_whitelist(emails)
    print(f"Added '{email}' to whitelist.")


def cmd_remove(args):
    emails = load_whitelist()
    email = args.email.lower()
    if email not in emails:
        print(f"'{email}' is not in the whitelist.")
        return
    emails.remove(email)
    save_whitelist(emails)
    print(f"Removed '{email}' from whitelist.")


def cmd_list(args):
    emails = load_whitelist()
    if not emails:
        print("Whitelist is empty.")
        return
    print("Whitelisted emails:")
    for email in sorted(emails):
        print(f"  - {email}")


def main():
    parser = argparse.ArgumentParser(description="Manage whitelisted emails for Depression Detector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add an email to the whitelist")
    add_parser.add_argument("email", help="Email address to add")
    add_parser.set_defaults(func=cmd_add)

    remove_parser = subparsers.add_parser("remove", help="Remove an email from the whitelist")
    remove_parser.add_argument("email", help="Email address to remove")
    remove_parser.set_defaults(func=cmd_remove)

    list_parser = subparsers.add_parser("list", help="List all whitelisted emails")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
