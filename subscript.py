#!/usr/bin/env python3
import sys
import os

# Add src to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from subscript.__main__ import main

if __name__ == "__main__":
    main()
