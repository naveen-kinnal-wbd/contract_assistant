"""
Blueprint Manager pane component
"""
import streamlit as st


def render_blueprint_manager():
    """Render the Blueprint Manager pane - currently placeholder"""
    st.markdown("### Blueprint Manager")
    
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 20px;
            background-color: #FAFAFA;
            border-radius: 8px;
            border: 1px dashed #E0E0E0;
            margin-top: 20px;
        ">
            <div style="
                font-size: 48px;
                margin-bottom: 16px;
            ">🔧</div>
            <h3 style="
                color: #616161;
                margin: 0 0 8px 0;
                font-weight: 500;
            ">Coming Soon</h3>
            <p style="
                color: #9E9E9E;
                margin: 0;
                text-align: center;
                font-size: 14px;
            ">
                The Blueprint Manager will allow you to view and manage<br>
                all blueprints used during metadata extraction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

