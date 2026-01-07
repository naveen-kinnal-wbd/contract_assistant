"""
Contract Assistance - Streamlit Frontend Application
"""

import streamlit as st
from services.api_client import ContractAPIClient
from components.contract_manager import render_contract_manager
from components.blueprint_manager import render_blueprint_manager


@st.cache_data(ttl=30)
def check_api_health(_api_client: ContractAPIClient) -> bool:
    """Cached health check to avoid blocking UI on every render"""
    try:
        return _api_client.health_check()
    except Exception:
        return False


# Page configuration
st.set_page_config(
    page_title="Contract Assistance",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for AWS-like simple styling
st.markdown(
    """
    <style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F2F0EF;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding-top: 0;
    }
    
    /* Header styling */
    .main-header {
        font-size: 24px;
        font-weight: 600;
        color: #232F3E;
        margin-bottom: 4px;
        padding: 0;
    }
    
    .main-subtitle {
        font-size: 14px;
        color: #687078;
        margin-bottom: 20px;
    }
    
    /* Navigation radio buttons */
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #232F3E;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 4px;
        transition: background-color 0.15s ease;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background-color: #E8E6E5;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #232F3E;
        color: #FFFFFF;
        border: none;
        font-weight: 500;
        padding: 8px 24px;
        border-radius: 4px;
        transition: background-color 0.15s ease;
    }
    
    .stButton > button:hover {
        background-color: #37475A;
        color: #FFFFFF;
        border: none;
    }
    
    .stButton > button:disabled {
        background-color: #D5D9D9;
        color: #879596;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #FF9900;
        color: #232F3E;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #EC7211;
        color: #232F3E;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border: 1px solid #D5D9D9;
        border-radius: 4px;
        padding: 8px 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007DBC;
        box-shadow: 0 0 0 1px #007DBC;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border: 1px solid #D5D9D9;
        border-radius: 4px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #D5D9D9;
        border-radius: 4px;
        padding: 16px;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #007DBC;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #E8E6E5;
        border-radius: 8px;
        background-color: #FFFFFF;
    }
    
    [data-testid="stExpander"] summary {
        font-weight: 500;
        color: #232F3E;
    }
    
    /* Toggle switch */
    [data-testid="stToggle"] {
        padding: 4px 0;
    }
    
    /* Divider */
    hr {
        border-color: #E8E6E5;
        margin: 16px 0;
    }
    
    /* Info/warning/error messages */
    .stAlert {
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 18px;
        font-weight: 600;
        color: #232F3E;
        padding: 16px 0 8px 0;
        border-bottom: 1px solid #D5D9D9;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_api_client() -> ContractAPIClient:
    """Initialize the API client"""
    return ContractAPIClient()


def main():
    """Main application entry point"""
    # Initialize API client
    api_client = init_api_client()

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-header">📄 Contract Assistance</div>',
            unsafe_allow_html=True,
        )

        # Navigation
        selected_pane = st.radio(
            "Navigation",
            options=["Contracts Manager", "Blueprint Manager"],
            index=0,
            label_visibility="collapsed",
        )

        # API status indicator
        st.markdown("---")
        try:
            api_healthy = check_api_health(api_client)
            if api_healthy:
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #4CAF50;"></span>
                        <span style="font-size: 12px; color: #687078;">API Connected</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #F44336;"></span>
                        <span style="font-size: 12px; color: #687078;">API Disconnected</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 8px; padding: 8px 0;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #FF9800;"></span>
                    <span style="font-size: 12px; color: #687078;">API Status Unknown</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Main content area
    if selected_pane == "Contracts Manager":
        st.markdown(
            '<h1 class="main-header">Contracts Manager</h1>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="main-subtitle">Upload contracts and track metadata extraction progress</p>',
            unsafe_allow_html=True,
        )
        render_contract_manager(api_client)

    elif selected_pane == "Blueprint Manager":
        st.markdown(
            '<h1 class="main-header">Blueprint Manager</h1>', unsafe_allow_html=True
        )
        st.markdown(
            '<p class="main-subtitle">Manage and view extraction blueprints</p>',
            unsafe_allow_html=True,
        )
        render_blueprint_manager()


if __name__ == "__main__":
    main()
