# Logging Configuration Guide

## Overview

AION Analytics uses a unified logging system with configurable log levels. By default, only WARNING and ERROR messages are shown in production to keep logs clean and focused on actual issues.

## Log Levels

The system supports four log levels (in order of increasing severity):

1. **DEBUG** (10) - Detailed operational information for debugging
2. **INFO** (20) - General informational messages
3. **WARNING** (30) - Warning messages indicating potential issues
4. **ERROR** (40) - Error messages indicating actual problems

## Configuration

### Environment Variable

Set the `LOG_LEVEL` environment variable to control which logs are shown:

```bash
# Production (default) - Only warnings and errors
export LOG_LEVEL=WARNING
python run_backend.py
python run_dt_backend.py

# Development - Show info, warnings, and errors (hide debug)
export LOG_LEVEL=INFO
python run_backend.py

# Debug - Show all logs including detailed operational info
export LOG_LEVEL=DEBUG
python run_backend.py

# Critical only - Only show error messages
export LOG_LEVEL=ERROR
python run_backend.py
```

### Default Behavior

If `LOG_LEVEL` is not set, the system defaults to **WARNING**, which means:
- ✅ WARNING messages are shown
- ✅ ERROR messages are shown
- ❌ INFO messages are hidden
- ❌ DEBUG messages are hidden

## What Gets Hidden by Default

### File Locking Logs (dt_backend)
Previously cluttered logs with 50+ lines per request:
```
[file_locking] ✅ Acquired exclusive lock: dt_trades.jsonl
[file_locking] 🔓 Released lock: dt_trades.jsonl
[file_locking] ✅ Acquired exclusive lock: dt_trades.jsonl
[file_locking] 🔓 Released lock: dt_trades.jsonl
... (40+ more lines)
```

**Now:** These are DEBUG level and hidden by default. Errors/timeouts still shown as WARNING.

### DT Execution Logs
Previously cluttered logs with operational details:
```
[dt_exec] 📝 Decision recording started for cycle abc123
[dt_exec] 📝 Phase 1 (pending): exec_001 BUY 100 AAPL
[dt_exec] ✅ Phase 2 (confirmed): exec_001 filled @ 150.25
[dt_exec] 🎉 Phase 3 (recorded): exec_001 complete
[dt_exec] 💾 saved rolling cache with position updates
```

**Now:** These are DEBUG level and hidden by default. Errors still shown.

### Bots Bundle Fetching Logs
Previously cluttered logs with 15+ timing lines per request:
```
[bots_page_bundle] Starting bundle fetch at 2026-01-19T09:35:20
[bots_page_bundle] Fetching swing/eod_status...
[bots_page_bundle] ✓ swing/eod_status completed in 14661.0ms
[bots_page_bundle] Fetching swing/configs...
[bots_page_bundle] ✓ swing/configs completed in 2.1ms
... (10+ more lines)
[bots_page_bundle] ✅ Bundle complete in 14671.8ms
```

**Now:** These are DEBUG level and hidden by default. Errors still shown as ERROR.

## What Still Shows by Default

Even with default WARNING level, you'll still see:

- ⚠️  **Warnings** - Potential issues that need attention
- ❌ **Errors** - Actual problems that failed
- 🚨 **Lock timeouts** - File locking conflicts
- 💥 **Execution failures** - Trade execution errors
- 🔥 **API failures** - HTTP errors, connection issues

## Usage Examples

### Production Deployment
```bash
# Quiet operation - only real problems shown
python run_backend.py
python run_dt_backend.py
```

Expected output:
```
[Backend] ✅ Startup complete — ready for requests.
[WARNING] Swing bot sync failed - retrying...
[ERROR] Unable to connect to database
```

### Development with Debug Info
```bash
# See all operational details
LOG_LEVEL=DEBUG python run_backend.py
LOG_LEVEL=DEBUG python run_dt_backend.py
```

Expected output:
```
[Backend] ✅ Startup complete — ready for requests.
[DEBUG] [file_locking] ✅ Acquired exclusive lock: dt_trades.jsonl
[DEBUG] [file_locking] 🔓 Released lock: dt_trades.jsonl
[DEBUG] [dt_exec] 📝 Decision recording started for cycle abc123
[DEBUG] [bots_page_bundle] Starting bundle fetch...
[DEBUG] [bots_page_bundle] ✓ swing/eod_status completed in 14661.0ms
```

### Troubleshooting a Specific Issue
```bash
# Show INFO level to see more context without full DEBUG noise
LOG_LEVEL=INFO python run_backend.py
```

## Code Usage

### Using the Logger

```python
from utils.logger import Logger, DEBUG, INFO, WARNING, ERROR

# Create a logger
logger = Logger(name="my_component", source="backend")

# Log at different levels
logger.debug("Detailed debug info")  # Hidden by default
logger.info("General information")   # Hidden by default
logger.warn("Warning message")       # Shown by default
logger.error("Error message")        # Shown by default

# Create a logger with custom log level
debug_logger = Logger(name="debug_component", source="backend", log_level=DEBUG)
debug_logger.debug("This will always show for this logger instance")
```

### Module-Level Functions

```python
from utils.logger import debug, log, warn, error

debug("Debug message")   # Hidden by default
log("Info message")      # Hidden by default (log = info)
warn("Warning message")  # Shown by default
error("Error message")   # Shown by default
```

### DT Backend Compatibility

```python
from dt_backend.core.logger_dt import debug, log, warn, error

# Works the same as utils.logger
debug("DT debug message")
log("DT info message")
warn("DT warning")
error("DT error")
```

## Benefits

✅ **Clean Production Logs** - Only see warnings and errors by default
✅ **Flexible Debugging** - Enable DEBUG when you need operational details
✅ **Better Signal-to-Noise** - Real issues stand out instead of drowning in noise
✅ **Performance** - Reduced I/O from fewer log writes in production
✅ **Easy Troubleshooting** - Set LOG_LEVEL=DEBUG temporarily to investigate issues

## Migration Notes

If you're maintaining or updating code:

- Replace `print()` statements with `logger.debug()` for operational info
- Replace `print()` for errors with `logger.error()`
- Use `logger.info()` for informational messages that should be visible at INFO level
- Use `logger.warn()` or `logger.warning()` for warnings
- Use `logger.error()` for errors

The old `log()` function is an alias for `info()` and will be hidden by default in production (WARNING level).
