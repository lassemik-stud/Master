from settings.logging import print_log

print_log.info('This is a test message')
print_log.debug('This is a test message')
print_log.warning('This is a test message')
print_log.error('This is a test message')
try:
    1 / 0  # This will raise a ZeroDivisionError
except Exception:
    print_log.error_exception('This is a test message')
print_log.critical('This is a test message')