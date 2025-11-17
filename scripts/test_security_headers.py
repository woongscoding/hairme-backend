#!/usr/bin/env python3
"""
Security Headers Test Script

Tests that all required security headers are present in HTTP responses.
"""

import sys
import subprocess
from typing import Dict, List, Tuple


def check_security_headers(url: str = "http://localhost:8000/api/health") -> Tuple[bool, List[str]]:
    """
    Check if security headers are present in the response

    Args:
        url: URL to test

    Returns:
        Tuple of (all_passed, failed_checks)
    """
    print(f"\n{'='*60}")
    print(f"🔍 Testing Security Headers")
    print(f"URL: {url}")
    print(f"{'='*60}\n")

    required_headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src",  # Check if CSP exists
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=",
    }

    optional_headers = {
        "Strict-Transport-Security": "max-age",  # Only for HTTPS
    }

    try:
        # Use curl to fetch headers
        result = subprocess.run(
            ["curl", "-s", "-I", url],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"❌ Failed to fetch URL: {result.stderr}")
            return False, ["Failed to fetch URL"]

        # Parse headers
        headers_raw = result.stdout
        headers: Dict[str, str] = {}

        for line in headers_raw.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()

        # Check required headers
        failed_checks = []
        passed_checks = []

        for header_name, expected_value in required_headers.items():
            if header_name in headers:
                actual_value = headers[header_name]
                if expected_value in actual_value:
                    print(f"✅ {header_name}: {actual_value[:50]}...")
                    passed_checks.append(header_name)
                else:
                    print(f"⚠️  {header_name}: {actual_value} (expected: {expected_value})")
                    failed_checks.append(f"{header_name} value mismatch")
            else:
                print(f"❌ {header_name}: MISSING")
                failed_checks.append(f"{header_name} missing")

        # Check optional headers
        print(f"\n{'='*60}")
        print("Optional Headers (HTTPS only):")
        print(f"{'='*60}\n")

        for header_name, expected_value in optional_headers.items():
            if header_name in headers:
                print(f"✅ {header_name}: {headers[header_name][:50]}...")
            else:
                print(f"ℹ️  {header_name}: Not set (OK for HTTP)")

        # Summary
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"{'='*60}")
        print(f"✅ Passed: {len(passed_checks)}/{len(required_headers)}")
        print(f"❌ Failed: {len(failed_checks)}/{len(required_headers)}")

        if failed_checks:
            print(f"\nFailed checks:")
            for check in failed_checks:
                print(f"  - {check}")

        all_passed = len(failed_checks) == 0
        return all_passed, failed_checks

    except subprocess.TimeoutExpired:
        print(f"❌ Request timed out")
        return False, ["Request timeout"]
    except FileNotFoundError:
        print(f"❌ curl command not found. Please install curl.")
        return False, ["curl not found"]
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False, [str(e)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test security headers")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/health",
        help="URL to test (default: http://localhost:8000/api/health)"
    )

    args = parser.parse_args()

    all_passed, failed_checks = check_security_headers(args.url)

    if all_passed:
        print(f"\n🎉 All security headers are properly configured!")
        sys.exit(0)
    else:
        print(f"\n❌ Security headers configuration needs improvement.")
        sys.exit(1)
