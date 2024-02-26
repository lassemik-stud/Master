from settings.logging import printLog as PrintLog

PrintLog.info('This is a test message')
PrintLog.debug('This is a test message')
PrintLog.warning('This is a test message')
PrintLog.error('This is a test message')
try:
    1 / 0  # This will raise a ZeroDivisionError
except Exception:
    PrintLog.error_exception('This is a test message')
PrintLog.critical('This is a test message')