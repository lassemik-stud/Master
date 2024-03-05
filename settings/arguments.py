#!/usr/bin/python3
# encoding: utf-8

import argparse

from settings.logging import printLog as PrintLog

def parse_arguments_controller(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-n1', '--message', help='this is the message', required=True)
    parser.add_argument('-n2', '--smessage', help='this is the message', required=True)
    args = parser.parse_args()

    PrintLog.debug('finished loading arguments')
    return args.message, args.smessage
