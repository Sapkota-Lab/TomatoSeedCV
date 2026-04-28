"""
Shiny app for strawberry/tomato seed detection and measurement.
Upload an image to segment seeds and view the mask and overlay results.
"""
import tempfile
from pathlib import Path
import io
import base64
import os

import cv2
import numpy as np
from shiny import App, render, ui, reactive
from dotenv import load_dotenv
load_dotenv()

from src.roboflow_rimdetect import run_rim_detection, summarize_rim
from src.whole_seed_roboflow import run_whole_seed_detection



# Helper function to convert cv2 image to base64 for display
def cv2_to_base64(image):
    """Convert OpenCV image to base64 string for HTML display."""
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"


IMAGE_STYLE = "max-width: 900px; width: 100%; height: auto; display: block; margin: 0 auto;"


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
            ui.input_radio_buttons(
                "seed_type",
                "Seed Type",
                choices={
                    "whole": "Whole Seed",
                    "bisected": "Bisected Seed (Rim)"
                },
                selected="whole"
            ),
            ui.input_numeric(
                "min_area_px",
                "Minimum Area (pixels)",
                value=20.0,
                min=1.0,
                max=1000.0,
                step=1.0
            ),
            ui.input_numeric(
                "min_area_mm2",
                "Minimum Area (mm²) — optional",
                value=2.0,
                min=0.1,
                max=100.0,
                step=0.1
            ),
            ui.input_numeric(
                "max_area_mm2",
                "Maximum Area (mm²) — optional",
                value=10.0,
                min=1.0,
                max=100.0,
                step=0.5
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
        seed_type = input.seed_type()
        if seed_type == "bisected":
            mm_per_pixel = .00811
        else:
            mm_per_pixel = input.mm_per_pixel()
        
        min_area_px = input.min_area_px()
        min_area_mm2 = input.min_area_mm2()
        max_area_mm2 = input.max_area_mm2()
        
        try:
            # Process the image based on seed type
            if seed_type == "whole":
                whole_output = run_whole_seed_detection(
                    file_path,
                    min_area_px=min_area_px,
                    mm_per_pixel=mm_per_pixel,
                    min_area_mm2=min_area_mm2,
                    max_area_mm2=max_area_mm2,
                )
                mask = whole_output["mask"]
                overlay = whole_output["overlay"]
                summary = whole_output["summary"]
                diagnostics = whole_output.get("diagnostics")

            elif seed_type == "bisected":
                # Run the Roboflow rim pipeline
                rim_output = run_rim_detection(file_path)
                mask = rim_output["mask"]
                overlay = rim_output["overlay"]
                summary = summarize_rim(mask, mm_per_pixel)
                diagnostics = None
        except Exception as exc:
            processed_data.set({"error": f"Processing failed: {exc}"})
            return
        
        # Store results
        processed_data.set({
            "original": image,
            "mask": mask,
            "overlay": overlay,
            "summary": summary,
            "seed_type": seed_type,
            "diagnostics": diagnostics,
            "error": None
        })
    @reactive.Effect
    def update_clibration_input():
        seed_type = input.seed_type()
        if seed_type == "bisected":
            ui.update_numeric("mm_per_pixel", value = .00811)
        if seed_type == "whole":
            ui.update_numeric("mm_per_pixel", value = 0.005341)
    
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
            f'<img src="{img_base64}" style="{IMAGE_STYLE}" />'        )
    
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
            f'<img src="{img_base64}" style="{IMAGE_STYLE}" />'        )
    
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
            f'<img src="{img_base64}" style="{IMAGE_STYLE}" />'        )
    
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
        
        seed_type = data.get("seed_type")
        summary = data["summary"]

        if summary is None:
            return ui.div(
                ui.p("No seeds detected."),
                style="padding: 20px; text-align: center; color: #666;"
            )
                
        # Get calibration value to determine units
        if seed_type == "bisected":
            mm_per_pixel = .00811
        else:
            mm_per_pixel = input.mm_per_pixel()
        has_calibration = mm_per_pixel is not None and mm_per_pixel > 0

        area_unit = "mm²" if has_calibration else "px²"
        thickness_unit = "mm" if has_calibration else "px"
        size_unit = "mm" if has_calibration else "px"
        
        if seed_type == "bisected":
            

            rim_area = summary["rim_area_mm2"] if has_calibration and summary["rim_area_mm2"] is not None else summary["rim_area_px"]
            avg_thickness = summary["avg_thickness_mm"] if has_calibration and summary["avg_thickness_mm"] is not None else summary["avg_thickness_px"]
            max_thickness = summary["max_thickness_mm"] if has_calibration and summary["max_thickness_mm"] is not None else summary["max_thickness_px"]
            min_thickness = summary["min_thickness_mm"] if has_calibration and summary["min_thickness_mm"] is not None else summary["min_thickness_px"]
            std_thickness = summary["std_thickness_mm"] if has_calibration and summary["std_thickness_mm"] is not None else summary["std_thickness_px"]

            calibration_note = (
            f'<p style="font-size: 10px; color: #080; font-weight: bold; margin-top: 10px;">'
            f'✓ Calibrated at {mm_per_pixel:.4f} mm/pixel'
            f'</p>'
            if has_calibration else
            '<p style="font-size: 10px; color: #c00; font-weight: bold; margin-top: 10px;">'
            'No mm calibration - values in pixels.'
            '</p>'
            )

            stats_html = f"""
            <style>
                .stats-table {{ font-family: monospace; font-size: 12px; overflow-x: auto; }}
                .stats-table table {{ border-collapse: collapse; width: 100%; max-width: 500px; }}
                .stats-table th {{ background-color: #f0f0f0; padding: 8px; text-align: left; border: 1px solid #ccc; }}
                .stats-table td {{ padding: 8px; border: 1px solid #e0e0e0; }}
            </style>
            <div class="stats-table">
                <p><strong>Bisected seed rim statistics</strong></p>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Rim Area</td>
                        <td>{rim_area:.4f} {area_unit}</td>
                    </tr>
                    <tr>
                        <td>Average Thickness</td>
                        <td>{avg_thickness:.4f} {thickness_unit}</td>
                    </tr>
                    <tr>
                        <td>Maximum Thickness</td>
                        <td>{max_thickness:.4f} {thickness_unit}</td>
                    </tr>
                    <tr>
                        <td>Minimum Thickness</td>
                        <td>{min_thickness:.4f} {thickness_unit}</td>
                    </tr>
                    <tr>
                        <td>Thickness Std. Dev.</td>
                        <td>{std_thickness:.4f} {thickness_unit}</td>
                    </tr>
                </table>
                {calibration_note}
            </div>
            """
            return ui.HTML(stats_html)
        
        else:
            diagnostics = data.get("diagnostics") or {}
            diagnostic_note = ""
            if diagnostics.get("area_filter_removed_all"):
                diagnostic_note = (
                    '<p style="font-size: 10px; color: #a60; font-weight: bold;">'
                    'Area filters removed every Roboflow detection, so the app is showing '
                    'the unfiltered whole-seed detections.'
                    '</p>'
                )
            elif diagnostics and diagnostics.get("prediction_count") == 0:
                diagnostic_note = (
                    '<p style="font-size: 10px; color: #c00; font-weight: bold;">'
                    'Roboflow returned 0 whole-seed predictions for this image.'
                    '</p>'
                )

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
                {}
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
            """.format(len(summary), diagnostic_note, unit_area=area_unit, unit_size=size_unit)
        
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
