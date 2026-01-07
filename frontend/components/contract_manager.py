"""
Contracts Manager pane component
"""

import base64
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import uuid
from typing import Optional
from services.api_client import (
    ContractAPIClient,
    DocumentGroup,
    DocumentMetadata,
    DocumentType,
    FileType,
    WorkflowStatus,
)
from .document_tile import render_document_tiles


def get_file_type_for_document(
    doc_type: DocumentType, is_master: bool = True
) -> FileType:
    """Determine file type based on document type and position"""
    if doc_type == DocumentType.STANDALONE:
        return FileType.STANDALONE
    elif doc_type == DocumentType.WAIVER:
        return FileType.WAIVER
    elif doc_type == DocumentType.MASTER:
        return FileType.MASTER if is_master else FileType.ATTACHMENT
    return FileType.STANDALONE


def render_contract_manager(api_client: ContractAPIClient):
    """Render the Contracts Manager pane"""
    st.markdown("### Contracts Manager")
    st.caption("Upload and process contract documents for metadata extraction")

    # Initialize session state for document groups
    if "document_groups" not in st.session_state:
        st.session_state.document_groups = []

    # Document upload form
    with st.container():
        # Document Type Selection
        doc_type_options = {
            "Standalone": DocumentType.STANDALONE,
            "Master": DocumentType.MASTER,
            "Waiver": DocumentType.WAIVER,
        }

        selected_doc_type = st.selectbox(
            "Document Type",
            options=list(doc_type_options.keys()),
            index=0,
            help="Select the type of contract document you are uploading",
        )
        doc_type = doc_type_options[selected_doc_type]

        # Main file upload (moved before Identifier Name)
        if doc_type == DocumentType.MASTER:
            st.markdown("##### Master Contract")
            master_files = st.file_uploader(
                "Upload Master Contract",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=False,
                key="master_upload",
                help="Upload the main master contract document",
            )

            # Attachments (only shown for Master type)
            st.markdown("##### Attachments (Optional)")
            attachment_files = st.file_uploader(
                "Add Attachments",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=True,
                key="attachment_upload",
                help="Upload any exhibits, schedules, or attachments",
            )

            uploaded_files = []
            if master_files:
                uploaded_files.append(("master", master_files))
            if attachment_files:
                for att in attachment_files:
                    uploaded_files.append(("attachment", att))
        else:
            # Single file upload for Standalone and Waiver
            uploaded_file = st.file_uploader(
                f"Upload {selected_doc_type} Contract",
                type=["pdf", "docx", "doc"],
                accept_multiple_files=False,
                key="single_upload",
                help=f"Upload the {selected_doc_type.lower()} contract document",
            )
            uploaded_files = [("single", uploaded_file)] if uploaded_file else []

        # Identifier Name (mandatory) - moved after file upload
        identifier_name = st.text_input(
            "Identifier Name *",
            placeholder="Enter a nickname for this document group",
            help="A friendly name to identify this document group in the processing queue",
        )

        # Refine Blueprints Toggle
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                """
                <p style="margin: 0; font-size: 14px; color: #424242;">
                    <strong>Refine Blueprints</strong><br>
                    <span style="color: #757575; font-size: 12px;">
                        Enable to analyze and refine extraction blueprints
                    </span>
                </p>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            refine_blueprints = st.toggle(
                "Refine",
                value=True,
                key="refine_toggle",
                label_visibility="collapsed",
            )

        # Submit button
        st.markdown("")
        submit_disabled = not identifier_name or not uploaded_files

        if st.button(
            "Upload & Process",
            type="primary",
            disabled=submit_disabled,
            use_container_width=True,
        ):
            if not identifier_name:
                st.error("Please enter an Identifier Name")
            elif not uploaded_files:
                st.error("Please upload at least one document")
            else:
                # Create document group
                group_id = str(uuid.uuid4())[:8]

                # Build document metadata list
                documents = []
                for file_role, file_obj in uploaded_files:
                    if file_obj is None:
                        continue

                    if file_role == "master":
                        file_type = FileType.MASTER
                    elif file_role == "attachment":
                        file_type = FileType.ATTACHMENT
                    else:
                        file_type = get_file_type_for_document(doc_type)

                    # Read and base64 encode file content
                    file_bytes = file_obj.getvalue()
                    encoded_content = base64.b64encode(file_bytes).decode("utf-8")

                    documents.append(
                        DocumentMetadata(
                            filename=file_obj.name,
                            file_type=file_type,
                            size_bytes=file_obj.size,
                            content_type=file_obj.type,
                            content=encoded_content,
                        )
                    )

                if documents:
                    # Create document group
                    document_group = DocumentGroup(
                        group_id=group_id,
                        identifier_name=identifier_name,
                        document_type=doc_type,
                        documents=documents,
                        refine_blueprints=refine_blueprints,
                    )

                    # Send to appropriate endpoint
                    try:
                        if refine_blueprints:
                            response = api_client.process_blueprints_refinement(
                                document_group
                            )
                        else:
                            response = api_client.process_contract_inference(
                                document_group
                            )

                        # Add to session state for tracking
                        st.session_state.document_groups.insert(
                            0,
                            {
                                "group_id": group_id,
                                "identifier_name": identifier_name,
                                "document_type": doc_type.value,
                                "refine_blueprints": refine_blueprints,
                            },
                        )

                        st.success(f"Processing started for '{identifier_name}'")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Failed to start processing: {str(e)}")

        # Help text
        if submit_disabled:
            if not identifier_name:
                st.caption("⚠️ Please enter an Identifier Name to continue")
            elif not uploaded_files:
                st.caption("⚠️ Please upload a document to continue")

    # Document tiles section
    st.markdown("---")
    render_document_tiles(st.session_state.document_groups, api_client)

    # Auto-refresh when there are in-progress workflows (non-blocking)
    if st.session_state.document_groups:
        has_in_progress = False
        for group_info in st.session_state.document_groups:
            group_id = group_info.get("group_id")
            try:
                progress = api_client.get_workflow_progress(group_id)
                if progress and progress.get("current_status") in [
                    WorkflowStatus.IN_PROGRESS.value,
                    WorkflowStatus.AWAITING_FEEDBACK.value,
                ]:
                    has_in_progress = True
                    break
            except Exception:
                pass

        if has_in_progress:
            # Non-blocking auto-refresh using streamlit-autorefresh
            st.caption("🔄 Auto-refreshing workflow status...")
            st_autorefresh(interval=1500, limit=None, key="workflow_autorefresh")
