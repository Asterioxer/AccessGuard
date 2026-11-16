"""
Storage safety and disk space validation for dataset operations.
Prevents unbounded downloads from filling local storage.
"""

import shutil
import os
from typing import Tuple
from app.core.logging import logger

def check_storage_safety(
    target_dir: str,
    requested_mb: float = 250.0,
    min_free_gb: float = 2.0
) -> Tuple[bool, str]:
    """
    Checks whether target directory has sufficient free disk space to store requested data
    while preserving a minimum safety margin (min_free_gb).
    
    Args:
        target_dir: Filesystem path to destination folder.
        requested_mb: Estimated size of incoming download in megabytes.
        min_free_gb: Minimum free gigabytes required to remain untouched.
        
    Returns:
        (is_safe, reason_message)
    """
    os.makedirs(target_dir, exist_ok=True)
    try:
        total, used, free = shutil.disk_usage(target_dir)
        free_gb = free / (1024 ** 3)
        requested_gb = requested_mb / 1024.0
        
        if (free_gb - requested_gb) < min_free_gb:
            msg = (
                f"Storage safety check failed: Free space ({free_gb:.2f} GB) minus "
                f"requested download ({requested_gb:.2f} GB) would fall below safety limit "
                f"({min_free_gb:.2f} GB)."
            )
            logger.warning(msg)
            return False, msg
            
        return True, f"Storage safe: {free_gb:.2f} GB free, requested {requested_gb:.2f} GB."
    except Exception as e:
        logger.error(f"Error checking storage safety for {target_dir}: {e}")
        return False, f"Could not determine disk usage: {e}"
