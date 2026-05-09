#!/bin/bash

# Script to fetch fresh economic data from FRED

# Activate virtual environment
source env/bin/activate

# Run data fetcher
python data_fetcher.py
