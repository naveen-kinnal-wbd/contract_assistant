"""
Document Tile component for displaying workflow progress
"""

import html
import streamlit as st
from typing import Optional
from services.api_client import WorkflowStatus


def get_status_indicator(status: str) -> tuple[str, str]:
    """
    Get the status indicator color and label based on workflow status
    Returns: (color, label)
    """
    status_map = {
        WorkflowStatus.IN_PROGRESS.value: ("#2196F3", "In Progress"),
        WorkflowStatus.AWAITING_FEEDBACK.value: ("#FF9800", "Awaiting Feedback"),
        WorkflowStatus.COMPLETED.value: ("#4CAF50", "Completed"),
        WorkflowStatus.FAILED.value: ("#F44336", "Failed"),
    }
    return status_map.get(status, ("#9E9E9E", "Unknown"))


def render_status_badge(status: str):
    """Render a status badge with appropriate color"""
    color, label = get_status_indicator(status)
    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 6px;">
            <span style="
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: {color};
                display: inline-block;
                box-shadow: 0 0 4px {color};
            "></span>
            <span style="
                font-size: 13px;
                color: {color};
                font-weight: 500;
            ">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_step_color(status: str) -> str:
    """Get color for workflow step based on status"""
    color_map = {
        WorkflowStatus.COMPLETED.value: "#4CAF50",
        WorkflowStatus.FAILED.value: "#F44336",
        WorkflowStatus.AWAITING_FEEDBACK.value: "#FF9800",
        WorkflowStatus.IN_PROGRESS.value: "#2196F3",
    }
    return color_map.get(status, "#E0E0E0")


def get_step_icon(status: str) -> str:
    """Get icon for workflow step based on status"""
    icon_map = {
        WorkflowStatus.COMPLETED.value: "✓",
        WorkflowStatus.FAILED.value: "✗",
        WorkflowStatus.AWAITING_FEEDBACK.value: "⏸",
        WorkflowStatus.IN_PROGRESS.value: "●",
    }
    return icon_map.get(status, "○")


def render_asset_selection_table(group_id: str, selection_data: list[dict], api_client):
    """Render the asset selection table with Select buttons"""
    if not selection_data:
        return

    st.markdown("**Select an asset to continue:**")

    # Create a table-like display using columns
    # Header row
    cols = st.columns([2, 2, 1, 1, 1])
    cols[0].markdown("**Deal Name**")
    cols[1].markdown("**Asset Name**")
    cols[2].markdown("**Deal ID**")
    cols[3].markdown("**Asset ID**")
    cols[4].markdown("**Action**")

    st.markdown("---")

    # Data rows
    for idx, item in enumerate(selection_data):
        cols = st.columns([2, 2, 1, 1, 1])
        cols[0].write(item.get("deal_name", ""))
        cols[1].write(item.get("asset_name", ""))
        cols[2].write(item.get("deal_id", ""))
        cols[3].write(item.get("asset_id", ""))

        if cols[4].button("Select", key=f"select_{group_id}_{idx}"):
            try:
                api_client.select_asset(group_id=group_id, selection=item)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to submit selection: {str(e)}")


def render_program_selection_table(
    group_id: str, selection_data: list[dict], api_client
):
    """Render the program selection table with Select buttons"""
    if not selection_data:
        return

    st.markdown(
        "**The following programs have been extracted from the contract. "
        "Please select a program to continue:**"
    )

    # Create a table-like display using columns
    # Header row - adjust column widths for readability
    cols = st.columns([2, 1.5, 2, 2, 1.5, 1.5, 1])
    cols[0].markdown("**Program**")
    cols[1].markdown("**Type**")
    cols[2].markdown("**Contract Name**")
    cols[3].markdown("**Parties**")
    cols[4].markdown("**Effective**")
    cols[5].markdown("**Executed**")
    cols[6].markdown("**Action**")

    st.markdown("---")

    # Data rows
    for idx, item in enumerate(selection_data):
        cols = st.columns([2, 1.5, 2, 2, 1.5, 1.5, 1])

        # Program name
        program_name = item.get("program_name", "N/A")
        cols[0].write(program_name)

        # Contract type
        contract_type = item.get("contract_type", "")
        cols[1].write(contract_type or "-")

        # Contract name (truncate if too long)
        contract_name = item.get("contract_name", "")
        if contract_name and len(contract_name) > 30:
            contract_name = contract_name[:27] + "..."
        cols[2].write(contract_name or "-")

        # Parties (format as comma-separated list)
        parties = item.get("parties", [])
        if parties:
            party_names = []
            for p in parties:
                if isinstance(p, dict):
                    party_names.append(p.get("value", str(p)))
                else:
                    party_names.append(str(p))
            parties_str = ", ".join(party_names)
            if len(parties_str) > 30:
                parties_str = parties_str[:27] + "..."
        else:
            parties_str = "-"
        cols[3].write(parties_str)

        # Date effective
        date_effective = item.get("date_effective", "")
        cols[4].write(date_effective or "-")

        # Date executed
        date_executed = item.get("date_executed", "")
        cols[5].write(date_executed or "-")

        # Select button
        if cols[6].button("Select", key=f"select_program_{group_id}_{idx}"):
            try:
                api_client.select_program(group_id=group_id, selection=item)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to submit selection: {str(e)}")


def is_program_selection(selection_data: list[dict]) -> bool:
    """
    Determine if the selection_data is for program selection or asset selection.
    Program selection data has 'program_name' key.
    """
    if not selection_data:
        return False
    first_item = selection_data[0]
    return "program_name" in first_item


def render_workflow_steps(
    steps: list[dict], current_step_index: int, group_id: str = None, api_client=None
):
    """Render the workflow steps using native Streamlit components"""
    if not steps:
        st.caption("No workflow steps yet...")
        return

    for i, step in enumerate(steps):
        status = step.get("status", "")
        message = step.get("message", "Processing...")
        timestamp = step.get("timestamp", "")
        selection_data = step.get("selection_data")

        if timestamp:
            # Format timestamp for display
            timestamp = timestamp.replace("T", " ").split(".")[0]

        color = get_step_color(status)
        icon = get_step_icon(status)

        # Create columns for icon and content
        col1, col2 = st.columns([0.08, 0.92])

        with col1:
            # Render colored status icon
            st.markdown(
                f'<span style="color: {color}; font-size: 16px; font-weight: bold;">{icon}</span>',
                unsafe_allow_html=True,
            )

        with col2:
            # Render message and timestamp as plain text
            st.markdown(f"**{message}**")
            if timestamp:
                st.caption(timestamp)

        # Show selection table if this step requires feedback
        if (
            status == WorkflowStatus.AWAITING_FEEDBACK.value
            and selection_data
            and group_id
            and api_client
        ):
            st.markdown("")
            # Determine which type of selection table to render
            if is_program_selection(selection_data):
                render_program_selection_table(group_id, selection_data, api_client)
            else:
                render_asset_selection_table(group_id, selection_data, api_client)


def render_document_tile(
    group_id: str, identifier_name: str, progress: Optional[dict], api_client=None
):
    """Render a single document tile with expandable details"""
    status = (
        progress.get("current_status", WorkflowStatus.IN_PROGRESS.value)
        if progress
        else WorkflowStatus.IN_PROGRESS.value
    )
    color, label = get_status_indicator(status)

    # Escape user-provided content for safe HTML rendering
    safe_identifier_name = html.escape(identifier_name)
    safe_group_id = html.escape(group_id)

    # Create a tile container with custom HTML header
    tile_header_html = f"""
    <div class="document-tile" style="
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            width: 100%;
        ">
            <div style="display: flex; flex-direction: column; gap: 4px;">
                <span style="font-weight: 600; font-size: 15px; color: #232F3E;">📄 {safe_identifier_name}</span>
                <span style="font-size: 12px; color: #687078;">ID: {safe_group_id}</span>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background-color: {color}15; border-radius: 12px;">
                <span style="
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background-color: {color};
                    display: inline-block;
                    box-shadow: 0 0 4px {color};
                "></span>
                <span style="
                    font-size: 12px;
                    color: {color};
                    font-weight: 500;
                ">{label}</span>
            </div>
        </div>
    </div>
    """

    st.markdown(tile_header_html, unsafe_allow_html=True)

    # Determine if we should auto-expand (when awaiting feedback)
    should_expand = status == WorkflowStatus.AWAITING_FEEDBACK.value

    # Create the expander for workflow details below the header
    with st.expander("View Workflow Details", expanded=should_expand):
        # Workflow steps
        if progress:
            steps = progress.get("steps", [])
            current_step_index = progress.get("current_step_index", 0)
            render_workflow_steps(steps, current_step_index, group_id, api_client)
        else:
            st.caption("Initializing workflow...")

    # Custom CSS for expander styling
    st.markdown(
        """
        <style>
        [data-testid="stExpander"] {
            border: 1px solid #E0E0E0;
            border-radius: 8px;
            margin-top: -12px;
            margin-bottom: 16px;
        }
        [data-testid="stExpander"] summary {
            font-size: 13px;
            color: #687078;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_document_tiles(document_groups: list[dict], api_client):
    """
    Render all document tiles for tracking workflow progress

    Args:
        document_groups: List of document group info dicts with keys:
            - group_id: str
            - identifier_name: str
        api_client: ContractAPIClient instance for fetching progress
    """
    if not document_groups:
        st.info(
            "No documents uploaded yet. Upload documents above to start processing."
        )
        return

    st.markdown("### Processing Status")

    for group_info in document_groups:
        group_id = group_info.get("group_id")
        identifier_name = group_info.get("identifier_name", group_id)

        # Fetch current progress from API
        try:
            progress = api_client.get_workflow_progress(group_id)
        except Exception:
            progress = None

        render_document_tile(group_id, identifier_name, progress, api_client)
