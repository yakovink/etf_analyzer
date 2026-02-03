

import traceback
def show_full_traceback(type, value, tb):
    traceback.print_exception(type, value, tb, limit=None)
import sys
sys.excepthook = show_full_traceback #set working dir            ectory to parent folder

#set jupiter notebook to not shrink error stack traces
sys.tracebacklimit = None
import os
import urllib3
import requests
import pandas as pd
import eodhd
import numpy as np
import pycountry
import time
import sqlite3
import logging
import ssl
from typing import List, Dict, Optional, Any, Literal, Tuple
import zipfile
import io
import json
import torch
from pandas_datareader import wb
import pandas_datareader as pdr

import datetime

