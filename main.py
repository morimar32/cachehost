#!/usr/bin/env python3
"""CacheHost - backward-compatible entry point.

Delegates to cachehost.__main__.main() so that `python3 main.py` continues to work.
"""

from cachehost.__main__ import main

if __name__ == "__main__":
    main()
