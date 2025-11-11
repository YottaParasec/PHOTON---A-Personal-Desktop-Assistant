import sys
import os

# Add the photon_test directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'photon_test')))

# Now you can import photon.py from photon_test
import photon

if __name__ == "__main__":
    # The main loop is inside photon.py, so we don't need to do anything here
    pass
