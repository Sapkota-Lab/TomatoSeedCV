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
    ui.panel_title("Tomato Seed CV"),
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
            ),
            ui.nav_panel(
                "Statistics",
                ui.output_ui("statistics_panel")
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
    
    @output
    @render.ui
    def statistics_panel():
        """Display per-seed statistics for all detected seeds."""
        data = processed_data.get()
        if data is None or data.get("error"):
            return ui.div(
                ui.p("No statistics available. Process an image first."),
                style="padding: 20px; text-align: center; color: #666;"
            )
        
        summary = data["summary"]
        if not summary:
            return ui.div(
                ui.p("No seeds detected."),
                style="padding: 20px; text-align: center; color: #666;"
            )
        
        # Get calibration value to determine units
        mm_per_pixel = input.mm_per_pixel()
        has_calibration = mm_per_pixel is not None and mm_per_pixel > 0
        
        # Build per-seed HTML table
        area_unit = "mm²" if has_calibration else "px²"
        size_unit = "mm" if has_calibration else "px"
        
        stats_html = """
        <style>
            .stats-table {{ font-family: monospace; font-size: 11px; overflow-x: auto; }}
            .stats-table table {{ border-collapse: collapse; }}
            .stats-table th {{ background-color: #f0f0f0; padding: 8px; text-align: right; border: 1px solid #ccc; }}
            .stats-table td {{ padding: 8px; text-align: right; border: 1px solid #e0e0e0; }}
            .stats-table .seed-id {{ text-align: center; font-weight: bold; background-color: #f9f9f9; }}
        </style>
        <div class="stats-table">
            <p><strong>Per-seed statistics ({} seeds detected)</strong></p>
            <p style="font-size: 10px; color: #666;">Match the seed numbers on the overlay image to identify problematic detections.</p>
            <table>
                <tr style="background-color: #f0f0f0;">
                    <th class="seed-id">Seed #</th>
                    <th>Area ({unit_area})</th>
                    <th>Eq.Diam ({unit_size})</th>
                    <th>Perimeter ({unit_size})</th>
                    <th>AR</th>
                    <th>Circ</th>
                    <th>Elong</th>
                    <th>Compact</th>
                    <th>Round</th>
                </tr>
        """.format(len(summary), unit_area=area_unit, unit_size=size_unit)
        
        for idx, seed in enumerate(summary, start=1):
            if has_calibration:
                area_val = seed['area_mm2'] if seed['area_mm2'] is not None else seed['area_px']
                diam_val = seed['eq_diam_mm'] if seed['eq_diam_mm'] is not None else seed['eq_diam_px']
                perim_val = seed['perimeter_mm'] if seed['perimeter_mm'] is not None else seed['perimeter_px']
                area_fmt = f"{area_val:.3f}" if seed['area_mm2'] is not None else f"{area_val:.1f}"
                diam_fmt = f"{diam_val:.3f}" if seed['eq_diam_mm'] is not None else f"{diam_val:.2f}"
                perim_fmt = f"{perim_val:.3f}" if seed['perimeter_mm'] is not None else f"{perim_val:.2f}"
            else:
                area_fmt = f"{seed['area_px']:.1f}"
                diam_fmt = f"{seed['eq_diam_px']:.2f}"
                perim_fmt = f"{seed['perimeter_px']:.2f}"
            
            stats_html += f"""
                <tr>
                    <td class="seed-id">{idx}</td>
                    <td>{area_fmt}</td>
                    <td>{diam_fmt}</td>
                    <td>{perim_fmt}</td>
                    <td>{seed['aspect_ratio']:.3f}</td>
                    <td>{seed['circularity']:.3f}</td>
                    <td>{seed['elongation']:.3f}</td>
                    <td>{seed['compactness']:.3f}</td>
                    <td>{seed['roundness']:.3f}</td>
                </tr>
            """
        
        calibration_note = f"<p style=\"font-size: 10px; color: #080; font-weight: bold; margin-top: 10px;\">✓ Calibrated at {mm_per_pixel:.4f} mm/pixel - values in mm</p>" if has_calibration else "<p style=\"font-size: 10px; color: #c00; font-weight: bold; margin-top: 10px;\">⚠ No calibration - values in pixels. Set calibration value and reprocess.</p>"
        
        stats_html += """
            </table>
            {calibration_note}
            <p style="font-size: 10px; color: #666; margin-top: 15px;">
                <strong>Legend:</strong> AR=Aspect Ratio | Circ=Circularity | Elong=Elongation | Compact=Compactness | Round=Roundness
            </p>
        </div>
        """.format(calibration_note=calibration_note)
        return ui.HTML(stats_html)


app = App(app_ui, server)
