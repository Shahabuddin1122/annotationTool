import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
import zipfile
import io
import base64
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates

# Set page config for wide layout
st.set_page_config(
    page_title="Ball Annotation Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SuperAnnotate-like styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        border: 1px solid #e1e5e9;
        background-color: white;
        color: #364954;
        font-weight: 500;
    }

    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #4f46e5;
    }

    /* Active annotation mode button */
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }

    /* File uploader styling */
    .stFileUploader > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #d1d5db;
        border-radius: 8px;
    }

    /* Metrics styling */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* Annotation panel styling */
    .annotation-panel {
        background-color: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Canvas container */
    .canvas-container {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        background: white;
        padding: 10px;
        margin: 10px 0;
    }

    /* Success/Error alerts */
    .stAlert {
        border-radius: 6px;
    }

    /* Image annotation styles */
    .image-container {
        position: relative;
        display: inline-block;
        cursor: crosshair;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        background: white;
        padding: 10px;
        margin: 10px auto;
        display: block;
        text-align: center;
    }

    .clickable-image {
        cursor: crosshair;
        max-width: 100%;
        border-radius: 4px;
    }

    /* Annotation mode indicator */
    .annotation-mode-active {
        border: 3px solid #4f46e5 !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)


class StreamlitBallAnnotationTool:
    def __init__(self):
        # Initialize session state variables
        if 'image_list' not in st.session_state:
            st.session_state.image_list = []
        if 'current_image_index' not in st.session_state:
            st.session_state.current_image_index = 0
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 1.0
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = None
        if 'annotation_mode' not in st.session_state:
            st.session_state.annotation_mode = False
        if 'last_click_coordinates' not in st.session_state:
            st.session_state.last_click_coordinates = None
        if 'prevent_duplicate_clicks' not in st.session_state:
            st.session_state.prevent_duplicate_clicks = set()

    def load_image(self, image_file):
        """Load and process image file"""
        try:
            image = Image.open(image_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    def normalize_coordinates(self, x, y, img_width, img_height):
        """Convert pixel coordinates to normalized coordinates"""
        norm_x = x / img_width
        norm_y = y / img_height
        return norm_x, norm_y

    def denormalize_coordinates(self, norm_x, norm_y, img_width, img_height):
        """Convert normalized coordinates to pixel coordinates"""
        x = int(norm_x * img_width)
        y = int(norm_y * img_height)
        return x, y

    def save_annotations_as_yolo(self):
        """Generate YOLO format annotation files"""
        annotation_files = {}

        for filename, annotations in st.session_state.annotations.items():
            content = []
            for ann in annotations:
                # YOLO format: class_id x_center y_center width height
                # For point annotations, we use minimal width and height
                content.append(f"1 {ann['x']:.6f} {ann['y']:.6f} 0.1 0.1")

            annotation_files[f"{Path(filename).stem}.txt"] = "\n".join(content)

        return annotation_files

    def create_download_zip(self):
        """Create a ZIP file with all annotations"""
        annotation_files = self.save_annotations_as_yolo()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in annotation_files.items():
                zip_file.writestr(filename, content)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    def create_annotated_image(self, image, annotations):
        """Create image with annotation overlays"""
        img_height, img_width = image.shape[:2]
        annotated_image = image.copy()

        for i, ann in enumerate(annotations):
            # Convert normalized coordinates to pixel coordinates
            pixel_x, pixel_y = self.denormalize_coordinates(ann['x'], ann['y'], img_width, img_height)

            # Draw fixed 2px radius circle
            cv2.circle(annotated_image, (pixel_x, pixel_y), 20, (0, 0, 255), -1)

            # Add point number slightly offset to avoid overlapping with small circle
            cv2.putText(annotated_image, str(i + 1), (pixel_x + 8, pixel_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(annotated_image, str(i + 1), (pixel_x + 8, pixel_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return annotated_image

    def add_annotation_point(self, x, y, img_width, img_height, filename):
        """Add annotation point at given coordinates"""
        # Convert to normalized coordinates
        norm_x, norm_y = self.normalize_coordinates(x, y, img_width, img_height)

        # Initialize annotations list if not exists
        if filename not in st.session_state.annotations:
            st.session_state.annotations[filename] = []

        # Add the new annotation
        st.session_state.annotations[filename].append({
            'x': norm_x,
            'y': norm_y,
            'class_id': 0
        })

        return len(st.session_state.annotations[filename])

    def render_header(self):
        """Render the header section"""
        st.markdown("""
        <div class="header-container">
            <h1>⚽ Ball Annotation Tool</h1>
            <p>SuperAnnotate-style interface for ball position annotation</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.markdown("## 🎯 Project Controls")

            # File upload section
            st.markdown("### 📁 Upload Images")
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                accept_multiple_files=True,
                key="file_uploader"
            )

            if uploaded_files and uploaded_files != st.session_state.uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                st.session_state.image_list = uploaded_files
                st.session_state.current_image_index = 0
                st.session_state.annotations = {}
                st.success(f"✅ Loaded {len(uploaded_files)} images")
                st.rerun()

            # Navigation section
            if st.session_state.image_list:
                st.markdown("### 🧭 Navigation")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⬅️ Previous", disabled=st.session_state.current_image_index == 0):
                        st.session_state.current_image_index -= 1
                        st.session_state.annotation_mode = False
                        # Clear click tracking when changing images
                        st.session_state.prevent_duplicate_clicks.clear()
                        st.session_state.last_click_coordinates = None
                        st.rerun()

                with col2:
                    if st.button("➡️ Next",
                                 disabled=st.session_state.current_image_index >= len(st.session_state.image_list) - 1):
                        st.session_state.current_image_index += 1
                        st.session_state.annotation_mode = False
                        # Clear click tracking when changing images
                        st.session_state.prevent_duplicate_clicks.clear()
                        st.session_state.last_click_coordinates = None
                        st.rerun()

                # Image selector
                current_image_name = st.session_state.image_list[st.session_state.current_image_index].name
                image_names = [f.name for f in st.session_state.image_list]
                selected_index = st.selectbox(
                    "Select Image:",
                    range(len(image_names)),
                    format_func=lambda x: f"{x + 1}. {image_names[x]}",
                    index=st.session_state.current_image_index,
                    key="image_selector"
                )

                if selected_index != st.session_state.current_image_index:
                    st.session_state.current_image_index = selected_index
                    st.session_state.annotation_mode = False
                    # Clear click tracking when changing images
                    st.session_state.prevent_duplicate_clicks.clear()
                    st.session_state.last_click_coordinates = None
                    st.rerun()

            # Annotation controls
            if st.session_state.image_list:
                st.markdown("### ✏️ Annotation Controls")

                # Toggle annotation mode
                mode_button_text = "🛑 Exit Annotation Mode" if st.session_state.annotation_mode else "➕ Start Annotation Mode"
                mode_button_type = "secondary" if st.session_state.annotation_mode else "primary"

                if st.button(mode_button_text, type=mode_button_type):
                    st.session_state.annotation_mode = not st.session_state.annotation_mode
                    # Clear duplicate prevention when toggling mode
                    st.session_state.prevent_duplicate_clicks.clear()
                    st.session_state.last_click_coordinates = None
                    st.rerun()

                # Show current mode status
                if st.session_state.annotation_mode:
                    st.success("🎯 Annotation mode active - Click on the image to add points!")
                else:
                    st.info("💡 Click 'Start Annotation Mode' then click on ball centers in the image")

                # Clear current annotations
                if st.button("🗑️ Clear All Points"):
                    current_filename = st.session_state.image_list[st.session_state.current_image_index].name
                    st.session_state.annotations[current_filename] = []
                    st.rerun()

                # Undo last annotation
                if st.button("↩️ Undo Last Point"):
                    current_filename = st.session_state.image_list[st.session_state.current_image_index].name
                    if current_filename in st.session_state.annotations and st.session_state.annotations[
                        current_filename]:
                        st.session_state.annotations[current_filename].pop()
                        st.rerun()

            # Export section
            if st.session_state.annotations:
                st.markdown("### 📤 Export Annotations")

                total_annotations = sum(len(anns) for anns in st.session_state.annotations.values())
                st.info(f"📊 Total annotations: {total_annotations}")

                if st.button("💾 Prepare Download"):
                    zip_data = self.create_download_zip()
                    st.download_button(
                        label="📥 Download ZIP",
                        data=zip_data,
                        file_name="ball_annotations.zip",
                        mime="application/zip"
                    )

            # Instructions
            st.markdown("### ℹ️ Instructions")
            st.markdown("""
            1. **Upload Images**: Use the file uploader above
            2. **Start Annotation Mode**: Click the annotation mode button
            3. **Click to Annotate**: Click directly on ball centers in the image
            4. **Navigate**: Use Previous/Next or the dropdown
            5. **Remove Points**: Use the remove buttons in the annotation list
            6. **Export**: Download annotations in YOLO format

            **Tips:**
            - Enable annotation mode first
            - Click once on each ball center to add a point
            - Each click creates exactly one 2px radius point
            - Points are numbered in order of creation
            - Red circles with white borders show your annotations
            - You can remove individual points using the list below
            """)

    def render_annotation_interface(self, current_image, current_file):
        """Render the interactive annotation interface"""
        img_height, img_width = current_image.shape[:2]
        current_annotations = st.session_state.annotations.get(current_file.name, [])

        # Calculate display size (max 800px width)
        max_width = 800
        if img_width > max_width:
            scale = max_width / img_width
            display_width = max_width
            display_height = int(img_height * scale)
        else:
            display_width = img_width
            display_height = img_height
            scale = 1.0

        st.markdown("### 🎯 Interactive Annotation")

        # Status display
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            mode_indicator = "🟢 ACTIVE" if st.session_state.annotation_mode else "🔴 INACTIVE"
            st.markdown(f"**Annotation Mode:** {mode_indicator}")

        with col2:
            st.markdown(f"**Current Points:** {len(current_annotations)}")

        with col3:
            if st.button("🔄 Refresh View"):
                st.rerun()

        # Create annotated image
        annotated_image = self.create_annotated_image(current_image, current_annotations)

        # Resize for display if necessary
        if scale != 1.0:
            annotated_image = cv2.resize(annotated_image, (display_width, display_height))

        # Display image with click detection
        container_class = "image-container annotation-mode-active" if st.session_state.annotation_mode else "image-container"

        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)

        # Convert image to PIL for streamlit
        annotated_pil = Image.fromarray(annotated_image)

        # Use streamlit_image_coordinates for click detection
        try:
            # Try to use streamlit-image-coordinates if available
            clicked_coords = streamlit_image_coordinates(
                annotated_pil,
                width=display_width,
                key=f"image_coords_{current_file.name}_{st.session_state.current_image_index}"
            )

            # Handle clicks - only add point if in annotation mode and coordinates are new
            if (clicked_coords is not None and
                    st.session_state.annotation_mode and
                    clicked_coords != st.session_state.last_click_coordinates):

                click_x = clicked_coords["x"]
                click_y = clicked_coords["y"]

                # Create unique identifier for this click
                click_id = f"{current_file.name}_{click_x}_{click_y}"

                # Only process if this exact click hasn't been processed before
                if click_id not in st.session_state.prevent_duplicate_clicks:
                    # Scale coordinates back to original image size
                    orig_x = int(click_x / scale)
                    orig_y = int(click_y / scale)

                    # Add annotation
                    point_count = self.add_annotation_point(orig_x, orig_y, img_width, img_height, current_file.name)

                    # Store this click to prevent duplicates
                    st.session_state.prevent_duplicate_clicks.add(click_id)
                    st.session_state.last_click_coordinates = clicked_coords

                    st.success(f"✅ Added point #{point_count} at ({orig_x}, {orig_y})")
                    st.rerun()

        except ImportError:
            # Fallback: Use regular image display with manual coordinates
            st.image(
                annotated_pil,
                width=display_width,
                caption=f"Image: {current_file.name} | Points: {len(current_annotations)} | Mode: {'Click to add points' if st.session_state.annotation_mode else 'Start annotation mode first'}"
            )

            # Show fallback manual input when in annotation mode
            if st.session_state.annotation_mode:
                st.warning(
                    "⚠️ Interactive clicking requires 'streamlit-image-coordinates' package. Using manual input instead.")

                with st.expander("🎯 Manual Point Entry", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        x_pixel = st.number_input(
                            f"X coordinate (0-{img_width})",
                            min_value=0,
                            max_value=img_width - 1,
                            value=img_width // 2,
                            key="manual_x_pixel"
                        )

                    with col2:
                        y_pixel = st.number_input(
                            f"Y coordinate (0-{img_height})",
                            min_value=0,
                            max_value=img_height - 1,
                            value=img_height // 2,
                            key="manual_y_pixel"
                        )

                    with col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Add Point", key="add_manual_point"):
                            point_count = self.add_annotation_point(x_pixel, y_pixel, img_width, img_height,
                                                                    current_file.name)
                            st.success(f"✅ Added point #{point_count} at ({x_pixel}, {y_pixel})")
                            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    def render_annotation_list(self, current_annotations, current_file, img_width, img_height):
        """Render the list of current annotations"""
        if current_annotations:
            st.markdown("### 📋 Current Annotations")

            # Create a more interactive annotation list
            for i, ann in enumerate(current_annotations):
                pixel_x, pixel_y = self.denormalize_coordinates(ann['x'], ann['y'], img_width, img_height)

                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])

                with col1:
                    st.write(f"**{i + 1}**")

                with col2:
                    st.write(f"Pixel: ({pixel_x}, {pixel_y})")

                with col3:
                    st.write(f"Norm: ({ann['x']:.4f}, {ann['y']:.4f})")

                with col4:
                    st.write("Ball")

                with col5:
                    if st.button("🗑️", key=f"remove_{i}", help="Remove this point"):
                        st.session_state.annotations[current_file.name].pop(i)
                        st.rerun()

                if i < len(current_annotations) - 1:
                    st.divider()
        else:
            st.info("No annotations yet. Start annotation mode and click on ball centers in the image!")

    def render_main_content(self):
        """Render the main annotation interface"""
        if not st.session_state.image_list:
            st.markdown("""
            <div style="text-align: center; padding: 4rem; background-color: #f8f9fa; border-radius: 8px;">
                <h2>👋 Welcome to Ball Annotation Tool</h2>
                <p>Upload images from the sidebar to get started!</p>
                <div style="font-size: 4rem; margin: 2rem 0;">📸</div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Get current image
        current_file = st.session_state.image_list[st.session_state.current_image_index]
        current_image = self.load_image(current_file)

        if current_image is None:
            st.error("❌ Failed to load image")
            return

        img_height, img_width = current_image.shape[:2]

        # Display image info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📸 Image</h4>
                <p>{st.session_state.current_image_index + 1} of {len(st.session_state.image_list)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>📏 Dimensions</h4>
                <p>{img_width} × {img_height}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            current_annotations = st.session_state.annotations.get(current_file.name, [])
            st.markdown(f"""
            <div class="metric-container">
                <h4>🎯 Annotations</h4>
                <p>{len(current_annotations)} points</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            progress = (st.session_state.current_image_index + 1) / len(st.session_state.image_list)
            st.markdown(f"""
            <div class="metric-container">
                <h4>📊 Progress</h4>
                <p>{progress:.1%} complete</p>
            </div>
            """, unsafe_allow_html=True)

        # Current filename
        st.markdown(f"**Current file:** `{current_file.name}`")

        # Render annotation interface
        self.render_annotation_interface(current_image, current_file)

        # Render annotation list
        self.render_annotation_list(current_annotations, current_file, img_width, img_height)

    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitBallAnnotationTool()
        app.run()
    except Exception as e:
        print("\n" + "=" * 60)
        print("⚠️  STREAMLIT RUNTIME ERROR")
        print("=" * 60)
        print("This application must be run using the Streamlit command:")
        print("\n   streamlit run manual_annotation.py\n")
        print("If you don't have Streamlit installed, install it with:")
        print("\n   pip install streamlit opencv-python pillow numpy streamlit-image-coordinates\n")
        print("=" * 60)
        print(f"Error details: {e}")
        return


if __name__ == "__main__":
    main()