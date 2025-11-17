#!/usr/bin/env python3
"""
Test Health Check Endpoint

Tests the enhanced /api/health endpoint locally or in production.
"""

import sys
import json
import argparse
from typing import Dict, Any


def test_health_endpoint(base_url: str = "http://localhost:8000", deep: bool = False) -> bool:
    """
    Test health check endpoint

    Args:
        base_url: Base URL of the API
        deep: Whether to run deep health check

    Returns:
        True if all checks passed, False otherwise
    """
    import requests

    url = f"{base_url}/api/health"
    if deep:
        url += "?deep=true"

    print("\n" + "="*60)
    print(f"🏥 Testing Health Check Endpoint")
    print(f"URL: {url}")
    print(f"Deep Check: {deep}")
    print("="*60 + "\n")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False

        data = response.json()

        # Print overall status
        status = data.get("status", "unknown")
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"

        print(f"{status_emoji} Overall Status: {status.upper()}")
        print(f"📦 Version: {data.get('version', 'unknown')}")
        print(f"🌍 Environment: {data.get('environment', 'unknown')}")
        print(f"⏱️  Check Duration: {data.get('check_duration_ms', 0)}ms")

        # Print startup services
        print(f"\n{'='*60}")
        print("Startup Services:")
        print(f"{'='*60}\n")

        startup = data.get("startup", {})

        print("Required Services:")
        for service, status in startup.get("required_services", {}).items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {service}: {status}")

        print("\nOptional Services:")
        for service, status in startup.get("optional_services", {}).items():
            status_icon = "✅" if status else "⚠️"
            print(f"  {status_icon} {service}: {status}")

        # Print real-time checks
        print(f"\n{'='*60}")
        print("Real-time Health Checks:")
        print(f"{'='*60}\n")

        checks = data.get("checks", {})

        # System metrics
        if "system" in checks:
            system = checks["system"]
            if "cpu" in system:
                cpu = system["cpu"]
                print(f"💻 CPU Usage: {cpu.get('percent', 0)}% (Cores: {cpu.get('count', 0)})")

            if "memory" in system:
                mem = system["memory"]
                print(f"🧠 Memory: {mem.get('used_mb', 0):.0f}MB / {mem.get('total_mb', 0):.0f}MB ({mem.get('percent', 0)}%)")

            if "disk" in system and "percent" in system["disk"]:
                disk = system["disk"]
                print(f"💾 Disk: {disk.get('used_gb', 0):.1f}GB / {disk.get('total_gb', 0):.1f}GB ({disk.get('percent', 0)}%)")

        # DynamoDB
        if "dynamodb" in checks:
            db = checks["dynamodb"]
            db_status = db.get("status", "unknown")
            if db_status == "healthy":
                print(f"✅ DynamoDB: {db_status} ({db.get('latency_ms', 0)}ms)")
            elif db_status == "skipped":
                print(f"⏭️  DynamoDB: {db.get('message', 'skipped')}")
            else:
                print(f"❌ DynamoDB: {db_status} - {db.get('error', 'unknown error')}")

        # Circuit Breaker
        if "circuit_breaker" in checks:
            cb = checks["circuit_breaker"]
            state = cb.get("state", "unknown")
            state_emoji = "✅" if state == "closed" else "⚠️" if state == "half_open" else "❌"
            print(f"{state_emoji} Circuit Breaker: {state} (Failures: {cb.get('fail_counter', 0)})")

        # Gemini API
        if "gemini_api" in checks:
            gemini = checks["gemini_api"]
            gemini_status = gemini.get("status", "unknown")
            if gemini_status == "healthy":
                print(f"✅ Gemini API: {gemini_status} ({gemini.get('latency_ms', 0)}ms)")
            elif gemini_status == "skipped":
                print(f"⏭️  Gemini API: {gemini.get('message', 'skipped')}")
            else:
                print(f"❌ Gemini API: {gemini_status} - {gemini.get('error', 'unknown error')}")

        # Summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")

        if status == "healthy":
            print("🎉 All systems operational!")
            return True
        elif status == "degraded":
            print("⚠️  Some services are degraded but API is functional")
            return True
        else:
            print("❌ Critical services are down")
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed: Cannot connect to {base_url}")
        print("   Make sure the server is running")
        return False

    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
        return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test health check endpoint")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep health check (includes Gemini API ping)"
    )

    args = parser.parse_args()

    success = test_health_endpoint(args.url, args.deep)

    sys.exit(0 if success else 1)
