

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
from typing import List, Dict, Optional, Any, Literal, Tuple, Union
import zipfile
import io
import json
import torch
import sdmx
from pandas_datareader import wb
import pandas_datareader as pdr
import warnings
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold

