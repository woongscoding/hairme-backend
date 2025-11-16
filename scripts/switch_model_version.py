"""
ML Model Version Switcher Script

사용법:
  python scripts/switch_model_version.py --env production
  python scripts/switch_model_version.py --env staging
  python scripts/switch_model_version.py --env archive/v3_2024-11-10
  python scripts/switch_model_version.py --list
  python scripts/switch_model_version.py --info production
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def list_available_versions():
    """List all available model versions"""
    models_path = get_project_root() / "models"

    print("\n=== Available Model Versions ===\n")

    # Production
    prod_metadata = models_path / "production" / "metadata.json"
    if prod_metadata.exists():
        with open(prod_metadata) as f:
            data = json.load(f)
        print(f"[PRODUCTION]:")
        print(f"   Version: {data.get('version', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Description: {data.get('description', 'N/A')}")
        if data.get('metrics'):
            print(f"   Metrics: val_loss={data['metrics'].get('final_val_loss', 'N/A')}, "
                  f"val_mae={data['metrics'].get('final_val_mae', 'N/A')}")
        print()

    # Staging
    staging_metadata = models_path / "staging" / "metadata.json"
    if staging_metadata.exists():
        with open(staging_metadata) as f:
            data = json.load(f)
        print(f"[STAGING]:")
        print(f"   Version: {data.get('version', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Description: {data.get('description', 'N/A')}")
        print()

    # Archived versions
    archive_path = models_path / "archive"
    if archive_path.exists():
        archived_versions = sorted([d for d in archive_path.iterdir() if d.is_dir()])
        if archived_versions:
            print("[ARCHIVED VERSIONS]:")
            for version_dir in archived_versions:
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        data = json.load(f)
                    print(f"   - archive/{version_dir.name}:")
                    print(f"     Version: {data.get('version', 'N/A')}")
                    print(f"     Created: {data.get('created_date', 'N/A')}")
                    print(f"     Archived: {data.get('archived_date', 'N/A')}")
                    print(f"     Reason: {data.get('reason_for_archival', 'N/A')}")
                else:
                    print(f"   - archive/{version_dir.name} (no metadata)")
            print()


def show_version_info(env: str):
    """Show detailed information about a specific version"""
    models_path = get_project_root() / "models"

    if env in ["production", "staging"]:
        metadata_file = models_path / env / "metadata.json"
    elif env.startswith("archive/"):
        metadata_file = models_path / env / "metadata.json"
    else:
        print(f"❌ Invalid environment: {env}")
        return

    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return

    with open(metadata_file) as f:
        data = json.load(f)

    print(f"\n=== Model Version Info: {env} ===\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()


def switch_version(target_env: str, dry_run: bool = False):
    """Switch to a different model version"""
    models_path = get_project_root() / "models"
    env_file = get_project_root() / ".env"

    # Validate target environment
    if target_env in ["production", "staging"]:
        target_path = models_path / target_env
    elif target_env.startswith("archive/"):
        target_path = models_path / target_env
    else:
        print(f"❌ Invalid environment: {target_env}")
        print("   Valid options: production, staging, archive/vX_YYYY-MM-DD")
        return False

    if not target_path.exists():
        print(f"❌ Target environment does not exist: {target_path}")
        return False

    # Check if model files exist
    if target_env in ["production", "staging"]:
        model_file = target_path / "model.pth"
    else:
        model_file = target_path / "model.pt"

    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False

    # Read current .env file
    env_content = []
    model_env_found = False

    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.readlines()

        # Update MODEL_ENVIRONMENT line if it exists
        for i, line in enumerate(env_content):
            if line.startswith('MODEL_ENVIRONMENT='):
                env_content[i] = f'MODEL_ENVIRONMENT={target_env}\n'
                model_env_found = True
                break

    # Add MODEL_ENVIRONMENT if not found
    if not model_env_found:
        env_content.append(f'MODEL_ENVIRONMENT={target_env}\n')

    # Show what will be changed
    print(f"\n=== Model Version Switch ===\n")
    print(f"Target Environment: {target_env}")
    print(f"Model File: {model_file}")

    # Show metadata if available
    metadata_file = target_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"Version: {metadata.get('version', 'N/A')}")
        print(f"Description: {metadata.get('description', 'N/A')}")

    if dry_run:
        print("\n[DRY RUN] Would update .env file with:")
        print(f"  MODEL_ENVIRONMENT={target_env}")
        return True

    # Write updated .env file
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(env_content)

    print(f"\n[SUCCESS] Switched to: {target_env}")
    print(f"   Updated: {env_file}")
    print("\n[NOTE] Restart the application for changes to take effect")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="ML Model Version Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available versions
  python scripts/switch_model_version.py --list

  # Show info about production version
  python scripts/switch_model_version.py --info production

  # Switch to production version
  python scripts/switch_model_version.py --env production

  # Switch to archived version (dry run)
  python scripts/switch_model_version.py --env archive/v3_2024-11-10 --dry-run
        """
    )

    parser.add_argument(
        '--env',
        type=str,
        help='Target environment (production, staging, or archive/vX_YYYY-MM-DD)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available model versions'
    )
    parser.add_argument(
        '--info',
        type=str,
        help='Show detailed information about a specific version'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making actual changes'
    )

    args = parser.parse_args()

    # Handle commands
    if args.list:
        list_available_versions()
    elif args.info:
        show_version_info(args.info)
    elif args.env:
        success = switch_version(args.env, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
