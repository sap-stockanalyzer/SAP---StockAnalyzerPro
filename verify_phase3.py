#!/usr/bin/env python3
"""
Phase 3 Verification Script

This script verifies that Phase 3 router consolidation was completed successfully.
Run this script to validate the backend configuration.

Usage:
    python verify_phase3.py

Expected Output:
    ✅ All checks pass
    ✅ 9 routers mounted
    ✅ No legacy data routers
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_phase3():
    """Verify Phase 3 completion."""
    print("=" * 70)
    print("PHASE 3 VERIFICATION SCRIPT")
    print("=" * 70)
    print()
    
    errors = []
    warnings = []
    
    # Test 1: Import backend service
    print("Test 1: Importing backend service...")
    try:
        from backend.backend_service import app, ROUTERS
        print("  ✅ Backend service imported successfully")
    except Exception as e:
        errors.append(f"Failed to import backend_service: {e}")
        print(f"  ❌ Failed to import: {e}")
        return False
    
    # Test 2: Check router count
    print("\nTest 2: Checking router count...")
    router_count = len(ROUTERS)
    if router_count == 9 or router_count == 10:  # 10 if testing_router available
        print(f"  ✅ Router count correct: {router_count} routers")
    else:
        errors.append(f"Expected 9-10 routers, got {router_count}")
        print(f"  ❌ Expected 9-10 routers, got {router_count}")
    
    # Test 3: Check consolidated routers
    print("\nTest 3: Checking consolidated routers...")
    prefixes = [getattr(r, 'prefix', '/') for r in ROUTERS]
    
    required_consolidated = ['/api/page', '/api/admin', '/api/settings']
    missing_consolidated = [p for p in required_consolidated if p not in prefixes]
    
    if not missing_consolidated:
        print("  ✅ All 3 consolidated routers present")
    else:
        errors.append(f"Missing consolidated routers: {missing_consolidated}")
        print(f"  ❌ Missing consolidated routers: {missing_consolidated}")
    
    # Test 4: Check for legacy data routers
    print("\nTest 4: Checking for legacy data routers...")
    legacy_data_routers = ['/api/bots/page', '/api/dashboard', '/api/portfolio']
    found_legacy = [p for p in legacy_data_routers if p in prefixes]
    
    if not found_legacy:
        print("  ✅ No legacy data routers found")
    else:
        errors.append(f"Legacy data routers still present: {found_legacy}")
        print(f"  ❌ Legacy data routers still present: {found_legacy}")
    
    # Test 5: Check backward compatibility routers
    print("\nTest 5: Checking backward compatibility routers...")
    compat_routers = ['/api/system', '/admin', '/admin/tools']
    found_compat = [p for p in compat_routers if p in prefixes]
    
    if len(found_compat) == 3:
        print("  ✅ All backward compatibility routers present")
    else:
        missing_compat = [p for p in compat_routers if p not in prefixes]
        warnings.append(f"Missing backward compat routers: {missing_compat}")
        print(f"  ⚠️ Missing backward compat routers: {missing_compat}")
    
    # Test 6: Check essential routers
    print("\nTest 6: Checking essential routers...")
    essential_routers = ['/api/events', '/api/cache']
    found_essential = [p for p in essential_routers if p in prefixes]
    
    if len(found_essential) == 2:
        print("  ✅ All essential routers present")
    else:
        missing_essential = [p for p in essential_routers if p not in prefixes]
        errors.append(f"Missing essential routers: {missing_essential}")
        print(f"  ❌ Missing essential routers: {missing_essential}")
    
    # Test 7: Display router list
    print("\nTest 7: Router configuration:")
    print("  " + "-" * 66)
    for i, r in enumerate(ROUTERS, 1):
        prefix = getattr(r, 'prefix', '/')
        tags = getattr(r, 'tags', [])
        tag_str = ', '.join(tags) if tags else 'no tags'
        print(f"  {i:2}. {prefix:30} ({tag_str})")
    print("  " + "-" * 66)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if errors:
        print(f"\n❌ FAILED - {len(errors)} error(s) found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ SUCCESS - All tests passed!")
    
    if warnings:
        print(f"\n⚠️ {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    
    print("\n" + "=" * 70)
    
    return len(errors) == 0


if __name__ == "__main__":
    try:
        success = verify_phase3()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
