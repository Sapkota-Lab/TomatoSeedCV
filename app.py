"""
Shiny app for strawberry seed detection and measurement.
Upload an image to segment seeds and view the mask and overlay results.
"""
import tempfile
from pathlib import Path
import io
import base64

import cv2
import numpy as np
from shiny import App, render, ui, reactive

from src.train_model import segment_seeds, summarize_seeds, annotate, describe


# Helper function to convert cv2 image to base64 for display
def cv2_to_base64(image):
    """Convert OpenCV image to base64 string for HTML display."""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"


app_ui = ui.page_fluid(
    ui.panel_title("Tomato Seed Analyzer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file(
                "file_upload",
                "Upload Image",
                accept=[".jpg", ".jpeg", ".png"],
                multiple=False
            ),
            ui.input_numeric(
                "mm_per_pixel",
                "Calibration (mm per pixel)",
                value=0.005341,
                min=0.0001,
                max=1.0,
                step=0.0001
            ),
            ui.input_numeric(
                "min_area_px",
                "Minimum Area (pixels)",
                value=20.0,
                min=1.0,
                max=1000.0,
                step=1.0
            ),
            ui.input_action_button(
                "process",
                "Process Image",
                class_="btn-primary"
            ),
            width=300
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Original",
                ui.output_ui("original_image")
            ),
            ui.nav_panel(
                "Mask",
                ui.output_ui("mask_image")
            ),
            ui.nav_panel(
                "Overlay",
                ui.output_ui("overlay_image")
            )
        )
    )
)


def server(input, output, session):
    # Reactive value to store processed results
    processed_data = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.process)
    def process_image():
        """Process the uploaded image when the button is clicked."""
        file_info = input.file_upload()
        if not file_info:
            return
        
        # Read the uploaded image
        file_path = file_info[0]["datapath"]
        image = cv2.imread(file_path)
        
        if image is None:
            processed_data.set({"error": "Could not read the uploaded image."})
            return
        
        # Get parameters
        mm_per_pixel = input.mm_per_pixel()
        min_area_px = input.min_area_px()
        
        # Process the image
        mask, seeds = segment_seeds(image, min_area_px=min_area_px)
        summary = summarize_seeds(seeds, mm_per_pixel=mm_per_pixel)
        overlay = annotate(image, summary, mm_per_pixel)
        
        # Store results
        processed_data.set({
            "original": image,
            "mask": mask,
            "overlay": overlay,
            "summary": summary,
            "error": None
        })
    
    @output
    @render.ui
    def original_image():
        """Display the original uploaded image."""
        data = processed_data.get()
        if data is None:
            return ui.div(
                ui.p("Upload an image and click 'Process Image' to begin."),
                style="padding: 20px; text-align: center; color: #666;"
            )
        
        if data.get("error"):
            return ui.div(
                ui.p(data["error"]),
                style="padding: 20px; text-align: center; color: red;"
            )
        
        img_base64 = cv2_to_base64(data["original"])
        return ui.HTML(
            f'<img src="{img_base64}" style="max-width: 100%; height: auto;" />'
        )
    
    @output
    @render.ui
    def mask_image():
        """Display the segmentation mask."""
        data = processed_data.get()
        if data is None or data.get("error"):
            return ui.div(
                ui.p("No mask available. Process an image first."),
                style="padding: 20px; text-align: center; color: #666;"
            )
        
        img_base64 = cv2_to_base64(data["mask"])
        return ui.HTML(
            f'<img src="{img_base64}" style="max-width: 100%; height: auto;" />'
        )
    
    @output
    @render.ui
    def overlay_image():
        """Display the annotated overlay image."""
        data = processed_data.get()
        if data is None or data.get("error"):
            return ui.div(
                ui.p("No overlay available. Process an image first."),
                style="padding: 20px; text-align: center; color: #666;"
            )
        
        img_base64 = cv2_to_base64(data["overlay"])
        return ui.HTML(
            f'<img src="{img_base64}" style="max-width: 100%; height: auto;" />'
        )


app = App(app_ui, server)
