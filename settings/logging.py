#!/usr/bin/python3
# encoding: utf-8

import logging
from colorama import Fore, Style

# Gets or createas a logger
myLogger = logging.getLogger(__name__)

# Set log level
myLogger.setLevel(logging.INFO and logging.DEBUG)

# Define file log format
logFormat = '%(asctime)s - pid: %(process)d - %(levelname)s -  %(message)s'
fileLogFormat = logging.Formatter(logFormat)

# Add basic handler
basicHandler = logging.StreamHandler()

# Create formatters
debugHandler = logging.Formatter(Fore.GREEN + '%(asctime)s'  + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTGREEN_EX + ' %(message)s' + Style.RESET_ALL)
infoHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.WHITE + ' %(message)s' + Style.RESET_ALL)
warningHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.YELLOW + ' %(message)s' + Style.RESET_ALL)
errorHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' +  Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.RED + ' %(message)s' + Style.RESET_ALL)
exceptionHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTRED_EX + ' %(message)s' + Style.RESET_ALL)

# Add file handler and set format
fileHandler = logging.FileHandler('settings/mainlog.log')
fileFormatter = logging.Formatter(logFormat)
fileHandler.setFormatter(fileFormatter)

# Add handlers to logger
myLogger.addHandler(fileHandler)
myLogger.addHandler(basicHandler)

class PrintLog:
    def __init__(self):
        self.fileHandler = fileHandler
        self.fileHandler.setFormatter(fileLogFormat)
        self.basicHandler = basicHandler
        myLogger.addHandler(self.fileHandler)
        myLogger.addHandler(self.basicHandler)

    def debug(self, message):
        self.basicHandler.setFormatter(debugHandler)
        myLogger.debug(message)

    def info(self, message):
        self.basicHandler.setFormatter(infoHandler)
        myLogger.info(message)

    def warning(self, message):
        self.basicHandler.setFormatter(warningHandler)
        myLogger.warning(message)

    def error(self, message):
        self.basicHandler.setFormatter(errorHandler)
        myLogger.error(message)

    def exception(self, message):
        self.basicHandler.setFormatter(exceptionHandler)
        myLogger.exception(message)