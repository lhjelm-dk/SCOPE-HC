import streamlit as st
import importlib.util
import sys
from pathlib import Path

st.set_page_config(page_title="SCOPE-HC", layout="wide")

# Hide any leftover navigation elements (just in case)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] nav {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# Custom page router - loads pages from _pages_disabled folder
def load_page(page_path: str):
    """Dynamically load and execute a page module."""
    try:
        # Normalize page path
        if page_path.startswith("pages/"):
            page_path = page_path.replace("pages/", "_pages_disabled/")
        elif not page_path.startswith("_pages_disabled/"):
            page_path = f"_pages_disabled/{page_path}"
        
        # Get the filename
        filename = Path(page_path).name
        if not filename.endswith(".py"):
            filename += ".py"
        
        # Full path to the page file
        file_path = Path("_pages_disabled") / filename
        
        if not file_path.exists():
            st.error(f"Page not found: {file_path}")
            return False
        
        # Load the module
        spec = importlib.util.spec_from_file_location(
            f"_pages_disabled.{filename[:-3]}",
            file_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"_pages_disabled.{filename[:-3]}"] = module
            spec.loader.exec_module(module)
            if hasattr(module, "main"):
                module.main()
                return True
            else:
                st.error(f"Page {filename} does not have a main() function")
                return False
    except Exception as e:
        st.error(f"Error loading page {page_path}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False
    return False

# Initialize navigation first
from scopehc.ui.common import init_theme, render_custom_navigation, render_sidebar_settings, render_color_legend

init_theme()
render_custom_navigation()
render_sidebar_settings()
# Color legend only (Help & Glossary moved to Help & Math page)
render_color_legend()

# Get current page from session state or default to Overview
current_page = st.session_state.get("current_page", "_pages_disabled/00_Overview.py")

# Load the current page
load_page(current_page)
