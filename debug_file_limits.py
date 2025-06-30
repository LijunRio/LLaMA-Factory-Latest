#!/usr/bin/env python3
"""
Debug script to check file descriptor limits and system resources.
"""

import resource
import os
import subprocess
import sys

def check_file_limits():
    """Check current file descriptor limits."""
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"File descriptor limits:")
        print(f"  Soft limit: {soft_limit}")
        print(f"  Hard limit: {hard_limit}")
        
        # Check current usage
        try:
            # Count open files for current process
            proc_fd_dir = f"/proc/{os.getpid()}/fd"
            if os.path.exists(proc_fd_dir):
                open_files = len(os.listdir(proc_fd_dir))
                print(f"  Currently open files: {open_files}")
            else:
                print("  Could not count open files")
        except Exception as e:
            print(f"  Error counting open files: {e}")
            
    except Exception as e:
        print(f"Error getting file limits: {e}")

def check_system_limits():
    """Check system-wide limits."""
    try:
        # Check system-wide open files
        with open("/proc/sys/fs/file-nr", "r") as f:
            line = f.read().strip()
            allocated, unused, max_files = map(int, line.split())
            print(f"System-wide file handles:")
            print(f"  Allocated: {allocated}")
            print(f"  Unused: {unused}")
            print(f"  Maximum: {max_files}")
    except Exception as e:
        print(f"Error getting system limits: {e}")

def try_increase_limit():
    """Try to increase file descriptor limit."""
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Try to set to a higher value
        new_soft = min(8192, hard_limit)
        if new_soft > soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))
            print(f"Successfully increased soft limit from {soft_limit} to {new_soft}")
        else:
            print(f"Soft limit {soft_limit} is already at or above target {new_soft}")
            
    except Exception as e:
        print(f"Error increasing limit: {e}")

def main():
    print("=== File Descriptor Limit Debug ===")
    print(f"Python version: {sys.version}")
    print(f"Process ID: {os.getpid()}")
    print()
    
    print("1. Current limits:")
    check_file_limits()
    print()
    
    print("2. System limits:")
    check_system_limits()
    print()
    
    print("3. Trying to increase limit:")
    try_increase_limit()
    print()
    
    print("4. Final limits:")
    check_file_limits()

if __name__ == "__main__":
    main()
