#!/usr/bin/python3
# encoding: utf-8

from logging import log
from subprocess import Popen, PIPE

from settings.logging import log_application

def get_full_path(path):
    return run_linux_command(f'readlink -f {path}',response=True)

def run_linux_command(command, response=False):
    
    output = Popen(command, shell=True, stdout=PIPE)
    if response==True: return output.communicate()[0].decode('utf-8')


def runWindowsCommandCMD(command, response=False):
    command = f"powershell.exe -command {command}"
    log_application('DEBUG',command)
    output = Popen(command, shell=True, stdout=PIPE)
    return output.communicate()[0].decode('utf-8')

def scp(inputFile, outputFolder, sshUser, sshDomain):
    command = f'scp {inputFile} {sshUser}@{sshDomain}:{outputFolder}' 
    run_linux_command(command, response=True)
    
    log_application('INFO',f'copied {inputFile} to {outputFolder}')
 
