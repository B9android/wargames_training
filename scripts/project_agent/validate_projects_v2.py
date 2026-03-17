#!/usr/bin/env python3
"""Validation script for GitHub Projects v2 integration.

Checks prerequisites and validates core abstraction layer before running full agent tests.
Usage: python scripts/project_agent/validate_projects_v2.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def check_env_vars() -> dict[str, str]:
    """Check required environment variables."""
    print("📋 Checking environment variables...")
    required = ["GITHUB_TOKEN", "REPO_NAME"]
    env = {}
    missing = []

    for var in required:
        val = os.environ.get(var)
        if val:
            env[var] = val
            print(f"  ✅ {var}: {val[:20]}..." if len(val) > 20 else f"  ✅ {var}: {val}")
        else:
            missing.append(var)
            print(f"  ❌ {var}: Not set")

    if missing:
        print(f"\n❌ Missing: {', '.join(missing)}")
        return {}

    return env


def check_python_dependencies() -> bool:
    """Check required Python packages."""
    print("\n📦 Checking Python dependencies...")
    required = ["github", "yaml"]

    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}: Not installed")
            all_ok = False

    if not all_ok:
        print(f"\n💡 Install with: pip install PyGithub PyYAML")

    return all_ok


def check_gh_cli() -> bool:
    """Check gh CLI is installed and authenticated."""
    print("\n🔧 Checking GitHub CLI...")

    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"  ✅ {result.stdout.strip()}")
        else:
            print(f"  ❌ gh CLI not working: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"  ❌ gh CLI not found")
        return False
    except Exception as exc:
        print(f"  ❌ Error checking gh CLI: {exc}")
        return False

    return True


def check_github_token_scope(token: str) -> bool:
    """Verify token has required scopes."""
    print("\n🔐 Checking GitHub token scopes...")

    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            env={**os.environ, "GITHUB_TOKEN": token},
            timeout=5,
        )

        if result.returncode == 0:
            if "repo" in result.stdout and "project" in result.stdout:
                print(f"  ✅ Token has required scopes")
                return True
            else:
                print(f"  ⚠️  Token may be missing scopes. Check:")
                print(result.stdout)
                return True  # Warning only
        else:
            print(f"  ❌ Token validation failed: {result.stderr}")
            return False
    except Exception as exc:
        print(f"  ❌ Error checking token: {exc}")
        return False


def check_repo_access(repo_name: str, token: str) -> bool:
    """Check access to specified repository."""
    print(f"\n📂 Checking repository access: {repo_name}...")

    try:
        result = subprocess.run(
            ["gh", "repo", "view", repo_name, "--json", "name"],
            capture_output=True,
            text=True,
            env={**os.environ, "GITHUB_TOKEN": token},
            timeout=5,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            print(f"  ✅ Repository found: {data['name']}")
            return True
        else:
            print(f"  ❌ Repository not accessible: {result.stderr}")
            return False
    except Exception as exc:
        print(f"  ❌ Error checking repository: {exc}")
        return False


def check_project_board(repo_name: str, token: str) -> bool:
    """Check for GitHub Projects v2 board on repository."""
    print(f"\n📊 Checking GitHub Projects v2 board...")

    try:
        result = subprocess.run(
            ["gh", "project", "list", "--repo", repo_name, "--format", "json"],
            capture_output=True,
            text=True,
            env={**os.environ, "GITHUB_TOKEN": token},
            timeout=10,
        )

        if result.returncode == 0:
            projects = json.loads(result.stdout)
            if projects:
                print(f"  ✅ Found {len(projects)} project(s):")
                for proj in projects:
                    print(f"     - {proj['title']} (ID: {proj['id']})")
                return True
            else:
                print(f"  ⚠️  No projects found on repository")
                print(f"     Create a project: gh project create --title='Main' --repo={repo_name}")
                return False
        else:
            print(f"  ⚠️  Could not list projects (may require --owner flag)")
            return True  # Warning only
    except Exception as exc:
        print(f"  ⚠️  Error checking projects: {exc}")
        return True  # Warning only


def check_project_fields(repo_name: str, token: str) -> bool:
    """Check for required custom fields on project board."""
    print(f"\n🏷️  Checking project custom fields...")

    required_fields = [
        "Version",
        "Sprint",
        "Status",
        "Story Points",
        "Experiment Status",
        "Git Commit",
        "W&B Run",
    ]

    try:
        # This requires a more complex GraphQL query; for now, just warn
        print(f"  📝 Required fields:")
        for field in required_fields:
            print(f"     - {field}")
        print(f"\n  ⚠️  Skipping field verification (requires manual GraphQL query)")
        print(f"     Verify in GitHub Projects v2 UI")
        return True
    except Exception as exc:
        print(f"  ⚠️  Error checking fields: {exc}")
        return True


def check_projects_v2_module(repo_name: str, token: str) -> bool:
    """Test core ProjectsV2Client initialization."""
    print(f"\n🔌 Testing ProjectsV2Client initialization...")

    try:
        # Add scripts/project_agent to path
        project_agent_dir = Path(__file__).parent
        sys.path.insert(0, str(project_agent_dir))

        from projects_v2 import ProjectsV2Client

        client = ProjectsV2Client(token, repo_name)
        print(f"  ✅ ProjectsV2Client initialized")

        # Try to get field IDs (this will cache them)
        try:
            fields = client.get_field_ids()
            found = list(fields.keys())
            print(f"  ✅ Field ID caching works: {len(found)} fields cached")
            if not found:
                print(f"     ⚠️  No fields cached (project may not have custom fields)")
            return True
        except Exception as exc:
            print(f"  ⚠️  Could not get field IDs: {exc}")
            print(f"     This may be expected if project board not set up")
            return True  # Warning

    except ImportError as exc:
        print(f"  ❌ Could not import ProjectsV2Client: {exc}")
        return False
    except Exception as exc:
        print(f"  ❌ Error initializing ProjectsV2Client: {exc}")
        return False


def main() -> int:
    """Run all validation checks."""
    print("=" * 60)
    print("GitHub Projects v2 Integration Validation")
    print("=" * 60)

    # Phase 1: Environment & CLI
    env = check_env_vars()
    if not env:
        return 1

    if not check_python_dependencies():
        return 1

    if not check_gh_cli():
        # gh CLI is optional for testing, but helpful
        print("  💡 Install at: https://cli.github.com")

    # Phase 2: GitHub Authentication
    token = env.get("GITHUB_TOKEN", "")
    repo_name = env.get("REPO_NAME", "")

    if not check_github_token_scope(token):
        return 1

    if not check_repo_access(repo_name, token):
        return 1

    # Phase 3: Project Board Setup
    check_project_board(repo_name, token)
    check_project_fields(repo_name, token)

    # Phase 4: Core Module
    if not check_projects_v2_module(repo_name, token):
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("✅ Validation complete!")
    print("=" * 60)
    print("""
Next steps:
1. Run individual agent tests from PROJECTS_V2_TESTING_GUIDE.md
2. Start with dry-run mode: DRY_RUN=true python scripts/project_agent/<agent>.py
3. Check logs in W&B for any errors
4. Review field mappings in common.py PROJECTS_V2_FIELDS

For help, see: docs/PROJECTS_V2_TESTING_GUIDE.md
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
