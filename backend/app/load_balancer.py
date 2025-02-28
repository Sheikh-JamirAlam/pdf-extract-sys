import psutil
import asyncio
import platform
from datetime import datetime

# Configuration
MAX_CONCURRENT_JOBS = 3
MAX_CPU_PERCENT = 85
MAX_MEMORY_PERCENT = 85
SYSTEM_STATS_INTERVAL = 5  # seconds

class SystemLoadTracker:
    def __init__(self):
        self.active_jobs = 0
        self.cpu_percent = 0
        self.memory_percent = 0
        self.last_updated = datetime.now()
        self.is_overloaded = False
    
    def update_stats(self):
        self.cpu_percent = psutil.cpu_percent(interval=0.5)
        self.memory_percent = psutil.virtual_memory().percent
        self.last_updated = datetime.now()
        
        # Determine if system is overloaded
        self.is_overloaded = (
            self.active_jobs >= MAX_CONCURRENT_JOBS or
            self.cpu_percent >= MAX_CPU_PERCENT or
            self.memory_percent >= MAX_MEMORY_PERCENT
        )
    
    def can_accept_job(self):
        # If stats are old, update them
        time_since_update = (datetime.now() - self.last_updated).total_seconds()
        if time_since_update > SYSTEM_STATS_INTERVAL:
            self.update_stats()
        
        return not self.is_overloaded
    
    def increment_jobs(self):
        self.active_jobs += 1
        
    def decrement_jobs(self):
        self.active_jobs = max(0, self.active_jobs - 1)

# Create a singleton instance
load_tracker = SystemLoadTracker()

async def monitor_system_resources():
    """Background task to periodically update system resource statistics"""
    while True:
        load_tracker.update_stats()
        await asyncio.sleep(SYSTEM_STATS_INTERVAL)