import asyncio
import sys

# Critical fix for Python 3.14: ensure loop is always available
_original_get_event_loop = asyncio.get_event_loop

def get_custom_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            return _original_get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

asyncio.get_event_loop = get_custom_loop

# Import streamlit after patch
import streamlit.web.cli as stcli
import os

def main():
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
