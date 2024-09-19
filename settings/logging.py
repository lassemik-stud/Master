import logging
from colorama import Fore, Style

import logging
from colorama import Fore, Style

class PrintLog:
    def __init__(self):
        # Gets or creates a logger
        self.myLogger = logging.getLogger(__name__)

        # Set log level - DEBUG < INFO < WARNING < ERROR < CRITICAL
        self.myLogger.setLevel(logging.DEBUG)

        # Define file log format
        logFormat = '%(asctime)s %(levelname)s %(message)s'
        self.fileLogFormat = logging.Formatter(logFormat)

        # Add basic handler
        self.basicHandler = logging.StreamHandler()

        # Create formatters
        self.debugHandler = logging.Formatter(Fore.GREEN + '%(asctime)s'  + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTGREEN_EX + ' %(message)s' + Style.RESET_ALL)
        self.infoHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.WHITE + ' %(message)s' + Style.RESET_ALL)
        self.warningHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.YELLOW + ' %(message)s' + Style.RESET_ALL)
        self.errorHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' +  Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.RED + ' %(message)s' + Style.RESET_ALL)
        self.exceptionHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.LIGHTRED_EX + ' %(message)s' + Style.RESET_ALL)
        self.criticalHandler = logging.Formatter(Fore.GREEN + '%(asctime)s' + Fore.LIGHTBLACK_EX + ' - %(levelname)s - ' + Fore.RED + ' %(message)s' + Style.RESET_ALL)

        # Add file handler and set format
        root = '/home/lasse'
        self.fileHandler = logging.FileHandler(f'{root}/Master/settings/mainlog.log')
        self.fileHandler.setFormatter(self.fileLogFormat)

        # Add handlers to logger
        self.myLogger.addHandler(self.fileHandler)
        self.myLogger.addHandler(self.basicHandler)

    def debug(self, message):
        self.basicHandler.setFormatter(self.debugHandler)
        self.myLogger.debug(message)

    def info(self, message):
        self.basicHandler.setFormatter(self.infoHandler)
        self.myLogger.info(message)

    def warning(self, message):
        self.basicHandler.setFormatter(self.warningHandler)
        self.myLogger.warning(message)

    def error(self, message):
        self.basicHandler.setFormatter(self.errorHandler)
        self.myLogger.error(message)

    def error_exception(self, message):
        self.basicHandler.setFormatter(self.exceptionHandler)
        self.myLogger.exception(message)
    
    def critical(self, message):
        self.basicHandler.setFormatter(self.criticalHandler)
        self.myLogger.critical(message)

printLog = PrintLog()