#!/bin/bash

poetry run python3 mdsist/evidently_app.py > /dev/null 2>&1 &

poetry run fastapi run mdsist/app.py --port 80
