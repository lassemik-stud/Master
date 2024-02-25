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
debugHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLUE_EX + ' - pid: %(process)d' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTGREEN_EX + ' %(message)s' + Style.RESET_ALL)
infoHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLUE_EX + ' - pid: %(process)d' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.WHITE + ' %(message)s' + Style.RESET_ALL)
warningHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLUE_EX + ' - pid: %(process)d' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.YELLOW + ' %(message)s' + Style.RESET_ALL)
errorHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLUE_EX + ' - pid: %(process)d' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.RED + ' %(message)s' + Style.RESET_ALL)
exceptionHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLUE_EX + ' - pid: %(process)d' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTRED_EX + ' %(message)s' + Style.RESET_ALL)

# Add file handler and set format
fileHandler = logging.FileHandler('settings/mainlog.log')
fileFormatter = logging.Formatter(logFormat)
fileHandler.setFormatter(fileFormatter)

# Add handlers to logger
myLogger.addHandler(fileHandler)
myLogger.addHandler(basicHandler)

class logApplicationCustomClass():
    def __init__(self, message):
        self.message = message

    def setFileFormatAs():
        fileHandler.setFormatter(fileLogFormat)

    def debug(self,message):
        self.message = message
        self.setFileFormatAs()
        basicHandler.setFormatter(debugHandler)
        myLogger.debug(message)
    
    def info(self,message):
        self.message = message
        self.setFileFormatAs()
        basicHandler.setFormatter(infoHandler)
        myLogger.info(message)

def print_log(type, message):
    """
    DEBUG/INFO/WARNING/ERROR/EXCEPTION
    """
    # set formatter for fileHandler (mainlog.log)
    fileHandler.setFormatter(fileLogFormat)

    # set formatter for basicHandler (terminal)
    if type == 'DEBUG': 
        basicHandler.setFormatter(debugHandler)
        myLogger.debug(message)
    elif type == 'INFO': 
        basicHandler.setFormatter(infoHandler)
        myLogger.info(message)
    elif type == 'WARNING': 
        basicHandler.setFormatter(warningHandler)
        myLogger.warning(message)
    elif type == 'ERROR': 
        basicHandler.setFormatter(errorHandler)
        myLogger.error(message)
    elif type == 'EXCEPTION':
        basicHandler.setFormatter(exceptionHandler)
        myLogger.exception(message)
    else: 
        myLogger.error('not a correct type')