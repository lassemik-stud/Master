#!/usr/bin/python3
# encoding: utf-8

from settings.logging import log_application

def test_logging():
    log_application('DEBUG','This is a debug message')
    log_application('INFO','This is a info message')
    log_application('WARNING','This is a warning message')
    log_application('ERROR','This is a error message')
    log_application('EXCEPTION','This is a exception message')