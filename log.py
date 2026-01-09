"""
Logging Configuration for Loan Approval Checker App
Logs to console and daily log file (app_YYYYMMDD.log)
"""

import logging
from datetime import datetime

# Suppress Streamlit warnings FIRST before any other imports
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.CRITICAL)

# Configure logger
logger = logging.getLogger("loan_app")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler - daily log file
log_filename = f"app_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False

