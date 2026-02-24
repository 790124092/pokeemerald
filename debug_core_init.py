import os
import sys

# Add local libs to path
sys.path.insert(0, os.path.abspath("pylibs"))

try:
    import mgba.core
    print("mgba.core imported")

    print("Attempting to load pokeemerald.gba")
    core = mgba.core.find("pokeemerald.gba")
    print(f"Core found: {core}")

    if core:
        # Try to access a property that would trigger a call
        # But core.init is what fails.
        # Let's try to manually inspect the C object if possible via cffi
        pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

