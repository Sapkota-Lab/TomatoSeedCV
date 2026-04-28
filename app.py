import cv2
import numpy as np
from shiny import *
from dotenv import load_dotenv
from src import *

load_dotenv()

diagnostics = None

def server(input_data, output, session):
    try:
        # Your main logic goes here, replacing all input calls with input_data
        # Example: output.result = input_data.some_function()
        pass
    except Exception as e:
        # Handle exceptions
        pass
