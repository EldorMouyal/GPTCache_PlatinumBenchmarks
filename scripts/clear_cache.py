#!/usr/bin/env python3
"""
Cache cleanup utility for LLMCache project.

Removes the centralized cache/ directory and all its contents.
Provides dry-run and confirmation options for safety.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def get_cache_dir() -> Path:
    """Get the cache directory path relative to the project root."""
    # Assume script is in scripts/ and project root is one level up
    project_root = Path(__file__).parent.parent
    return project_root / "cache"


def get_cache_info(cache_dir: Path) -> dict:
    """Get information about cache contents."""
    info = {
        "exists": cache_dir.exists(),
        "size_bytes": 0,
        "file_count": 0,
        "subdirs": []
    }
    
    if info["exists"]:
        for item in cache_dir.rglob("*"):
            if item.is_file():
                info["file_count"] += 1
                info["size_bytes"] += item.stat().st_size
            elif item.is_dir() and item.parent == cache_dir:
                info["subdirs"].append(item.name)
    
    return info


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def clear_cache(cache_dir: Path, dry_run: bool = False) -> bool:
    """
    Clear the cache directory.
    
    Args:
        cache_dir: Path to cache directory
        dry_run: If True, only show what would be deleted
    
    Returns:
        True if successful (or would be successful in dry-run)
    """
    if not cache_dir.exists():
        print("✅ Cache directory does not exist - nothing to clean.")
        return True
    
    info = get_cache_info(cache_dir)
    
    print(f"📁 Cache directory: {cache_dir}")
    print(f"📊 Contents: {info['file_count']} files, {format_size(info['size_bytes'])}")
    if info["subdirs"]:
        print(f"📂 Subdirectories: {', '.join(info['subdirs'])}")
    
    if dry_run:
        print("🔍 DRY RUN - Would delete the entire cache/ directory and all contents")
        return True
    
    try:
        shutil.rmtree(cache_dir)
        print("✅ Cache directory cleared successfully")
        return True
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Clear LLMCache project cache directory")
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--yes",
        action="store_true", 
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        print("✅ No cache directory found - nothing to clean.")
        return 0
    
    # Show what will be affected
    info = get_cache_info(cache_dir)
    print(f"📁 Found cache directory: {cache_dir}")
    print(f"📊 Contents: {info['file_count']} files, {format_size(info['size_bytes'])}")
    if info["subdirs"]:
        print(f"📂 Subdirectories: {', '.join(info['subdirs'])}")
    
    # Confirmation (unless --yes or --dry-run)
    if not args.dry_run and not args.yes:
        response = input("\n🗑️  Delete entire cache directory? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Cancelled")
            return 1
    
    # Perform the cleanup
    success = clear_cache(cache_dir, dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())