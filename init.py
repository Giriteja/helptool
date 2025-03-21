# In a file called init.py
import os

# Set required environment variables
os.environ["POPLAR_SDK_ENABLED"] = "true"
os.environ["POPTORCH_LOG_LEVEL"] = "ERR"  # Reduce log verbosity
