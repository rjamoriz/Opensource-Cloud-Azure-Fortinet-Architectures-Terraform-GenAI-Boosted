"""
Google Cloud Authentication Component for Streamlit
Handles GCP project setup and authentication within the app
"""

import streamlit as st
import json
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

def display_gcp_auth_setup():
    """Display Google Cloud authentication setup interface"""
    
    st.subheader("🔐 Google Cloud Platform Authentication Setup")
    
    # Check current authentication status
    auth_status = check_gcp_auth_status()
    
    if auth_status["authenticated"]:
        st.success(f"✅ Authenticated with GCP Project: **{auth_status['project_id']}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Change Authentication", key="gcp_change_auth_main"):
                st.session_state.show_gcp_auth = True
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Authentication", key="gcp_clear_auth_main"):
                clear_gcp_auth()
                st.success("Authentication cleared!")
                st.rerun()
        
        return auth_status
    else:
        st.warning("⚠️ Google Cloud Platform not configured")
        return setup_gcp_authentication()

def setup_gcp_authentication() -> Dict[str, any]:
    """Setup GCP authentication with multiple options"""
    
    st.markdown("### Choose your authentication method:")
    
    auth_method = st.radio(
        "Authentication Method",
        [
            "🔑 Service Account Key (JSON file)",
            "🌐 Project ID Only (uses default credentials)",
            "⚙️ Manual Configuration"
        ],
        key="gcp_auth_method_radio"
    )
    
    if auth_method == "🔑 Service Account Key (JSON file)":
        return setup_service_account_auth()
    elif auth_method == "🌐 Project ID Only (uses default credentials)":
        return setup_project_id_auth()
    else:
        return setup_manual_auth()

def setup_service_account_auth() -> Dict[str, any]:
    """Setup authentication using service account JSON file"""
    
    st.markdown("#### Upload Service Account JSON File")
    st.info("💡 This is the most secure method for production use")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose service account JSON file",
            type="json",
            help="Upload the JSON file downloaded from GCP Console",
            key="gcp_service_account_uploader"
        )
    
    with col2:
        st.markdown("**Required Roles:**")
        st.markdown("""
        - Compute Admin
        - Project Viewer
        - IAM Service Account User
        - Storage Admin
        """)
    
    if uploaded_file is not None:
        try:
            # Parse the JSON file
            credentials_data = json.load(uploaded_file)
            
            # Validate required fields
            required_fields = ["type", "project_id", "private_key_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in credentials_data]
            
            if missing_fields:
                st.error(f"❌ Invalid service account file. Missing fields: {', '.join(missing_fields)}")
                return {"authenticated": False}
            
            # Display project information
            project_id = credentials_data.get("project_id")
            client_email = credentials_data.get("client_email")
            
            st.success(f"✅ Valid service account file detected")
            st.info(f"**Project ID:** {project_id}")
            st.info(f"**Service Account:** {client_email}")
            
            if st.button("💾 Save and Use This Configuration", key="gcp_save_service_account"):
                # Save credentials securely
                saved = save_gcp_credentials(credentials_data, "service_account")
                
                if saved:
                    st.success("✅ Credentials saved successfully!")
                    st.session_state.gcp_project_id = project_id
                    st.session_state.gcp_auth_method = "service_account"
                    return {
                        "authenticated": True,
                        "project_id": project_id,
                        "auth_method": "service_account"
                    }
                else:
                    st.error("❌ Failed to save credentials")
                    return {"authenticated": False}
        
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please upload a valid service account key file.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    return {"authenticated": False}

def setup_project_id_auth() -> Dict[str, any]:
    """Setup authentication using project ID and default credentials"""
    
    st.markdown("#### Project ID Configuration")
    st.info("💡 This method uses your default Google Cloud credentials (from gcloud CLI)")
    
    project_id = st.text_input(
        "Google Cloud Project ID",
        placeholder="my-project-123",
        help="Your GCP project ID (not the project name)",
        key="gcp_project_id_main"
    )
    
    if project_id:
        # Validate project ID format
        if not is_valid_project_id(project_id):
            st.error("❌ Invalid project ID format. Use lowercase letters, numbers, and hyphens only.")
            return {"authenticated": False}
        
        st.success(f"✅ Project ID: **{project_id}**")
        
        # Check if gcloud is available
        if check_gcloud_available():
            st.info("✅ Google Cloud CLI detected")
        else:
            st.warning("⚠️ Google Cloud CLI not detected. You may need to install it or use service account authentication.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Configuration", key="gcp_save_project_config"):
                saved = save_gcp_project_config(project_id)
                if saved:
                    st.success("✅ Configuration saved!")
                    st.session_state.gcp_project_id = project_id
                    st.session_state.gcp_auth_method = "default"
                    return {
                        "authenticated": True,
                        "project_id": project_id,
                        "auth_method": "default"
                    }
        
        with col2:
            if st.button("🧪 Test Connection", key="gcp_test_connection"):
                with st.spinner("Testing GCP connection..."):
                    test_result = test_gcp_connection(project_id)
                    if test_result["success"]:
                        st.success(f"✅ {test_result['message']}")
                    else:
                        st.error(f"❌ {test_result['message']}")
    
    return {"authenticated": False}

def setup_manual_auth() -> Dict[str, any]:
    """Setup authentication with manual credential entry"""
    
    st.markdown("#### Manual Credential Configuration")
    st.warning("⚠️ Only use this for testing. Service account files are recommended for production.")
    
    with st.form("manual_gcp_config"):
        project_id = st.text_input("Project ID", placeholder="my-project-123", key="gcp_manual_project_id")
        
        st.markdown("**Service Account Details (Optional):**")
        client_email = st.text_input("Client Email", placeholder="service-account@project.iam.gserviceaccount.com", key="gcp_manual_client_email")
        private_key = st.text_area("Private Key", placeholder="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----", key="gcp_manual_private_key")
        
        region = st.selectbox("Default Region", [
            "us-central1", "us-west1", "us-east1", "us-east4",
            "europe-west1", "europe-west2", "europe-west3",
            "asia-southeast1", "asia-northeast1", "asia-south1"
        ], key="gcp_manual_region")
        
        zone = st.selectbox("Default Zone", [
            f"{region}-a", f"{region}-b", f"{region}-c"
        ], key="gcp_manual_zone")
        
        submitted = st.form_submit_button("💾 Save Manual Configuration")
        
        if submitted and project_id:
            config = {
                "project_id": project_id,
                "region": region,
                "zone": zone
            }
            
            if client_email and private_key:
                config.update({
                    "client_email": client_email,
                    "private_key": private_key
                })
            
            saved = save_gcp_manual_config(config)
            if saved:
                st.success("✅ Manual configuration saved!")
                st.session_state.gcp_project_id = project_id
                st.session_state.gcp_auth_method = "manual"
                return {
                    "authenticated": True,
                    "project_id": project_id,
                    "auth_method": "manual"
                }
    
    return {"authenticated": False}

def check_gcp_auth_status() -> Dict[str, any]:
    """Check current GCP authentication status"""
    
    # Check session state first
    if hasattr(st.session_state, 'gcp_project_id') and st.session_state.gcp_project_id:
        return {
            "authenticated": True,
            "project_id": st.session_state.gcp_project_id,
            "auth_method": getattr(st.session_state, 'gcp_auth_method', 'unknown')
        }
    
    # Check saved configuration
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            project_id = config.get('project_id')
            if project_id:
                # Update session state
                st.session_state.gcp_project_id = project_id
                st.session_state.gcp_auth_method = config.get('auth_method', 'unknown')
                
                return {
                    "authenticated": True,
                    "project_id": project_id,
                    "auth_method": config.get('auth_method', 'unknown')
                }
        except Exception as e:
            logger.error(f"Error reading GCP config: {e}")
    
    return {"authenticated": False}

def save_gcp_credentials(credentials_data: Dict, auth_method: str) -> bool:
    """Save GCP credentials securely"""
    
    try:
        # Create config directory
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save credentials file
        credentials_path = config_dir / "credentials.json"
        with open(credentials_path, 'w') as f:
            json.dump(credentials_data, f, indent=2)
        
        # Save configuration
        config = {
            "project_id": credentials_data.get("project_id"),
            "auth_method": auth_method,
            "credentials_path": str(credentials_path)
        }
        
        config_path = get_config_path()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving GCP credentials: {e}")
        return False

def save_gcp_project_config(project_id: str) -> bool:
    """Save GCP project configuration"""
    
    try:
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config = {
            "project_id": project_id,
            "auth_method": "default"
        }
        
        config_path = get_config_path()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving GCP config: {e}")
        return False

def save_gcp_manual_config(config: Dict) -> bool:
    """Save manual GCP configuration"""
    
    try:
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config["auth_method"] = "manual"
        
        config_path = get_config_path()
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving manual GCP config: {e}")
        return False

def clear_gcp_auth() -> bool:
    """Clear GCP authentication"""
    
    try:
        # Clear session state
        if hasattr(st.session_state, 'gcp_project_id'):
            del st.session_state.gcp_project_id
        if hasattr(st.session_state, 'gcp_auth_method'):
            del st.session_state.gcp_auth_method
        
        # Remove config files
        config_dir = get_config_dir()
        if config_dir.exists():
            for file_path in config_dir.glob("*"):
                file_path.unlink()
        
        # Clear environment variable
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
        return True
        
    except Exception as e:
        logger.error(f"Error clearing GCP auth: {e}")
        return False

def test_gcp_connection(project_id: str) -> Dict[str, any]:
    """Test GCP connection"""
    
    try:
        # Try to import Google Cloud libraries
        from google.cloud import resource_manager_v3
        from google.auth import default
        
        # Attempt to get default credentials
        credentials, detected_project = default()
        
        if detected_project:
            return {
                "success": True,
                "message": f"Connection successful! Detected project: {detected_project}"
            }
        else:
            return {
                "success": True,
                "message": "Credentials found, but no default project detected"
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Connection failed: {str(e)}"
        }

def is_valid_project_id(project_id: str) -> bool:
    """Validate GCP project ID format"""
    import re
    
    # GCP project ID rules:
    # - 6-30 characters
    # - lowercase letters, numbers, hyphens
    # - start with letter
    # - end with letter or number
    pattern = r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$'
    return bool(re.match(pattern, project_id))

def check_gcloud_available() -> bool:
    """Check if gcloud CLI is available"""
    try:
        import subprocess
        result = subprocess.run(
            ["gcloud", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def get_config_dir() -> Path:
    """Get configuration directory path"""
    home = Path.home()
    return home / ".fortigate-chatbot" / "gcp"

def get_config_path() -> Path:
    """Get configuration file path"""
    return get_config_dir() / "config.json"

def display_gcp_auth_status():
    """Display current GCP authentication status in sidebar or info box"""
    
    auth_status = check_gcp_auth_status()
    
    if auth_status["authenticated"]:
        st.success(f"🌐 GCP: {auth_status['project_id']}")
        return True
    else:
        st.warning("🌐 GCP: Not configured")
        return False

def get_gcp_credentials() -> Optional[str]:
    """Get the path to GCP credentials file"""
    
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('credentials_path')
        except:
            pass
    
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

def get_gcp_project_id() -> Optional[str]:
    """Get the configured GCP project ID"""
    
    # Check session state first
    if hasattr(st.session_state, 'gcp_project_id'):
        return st.session_state.gcp_project_id
    
    # Check saved configuration
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('project_id')
        except:
            pass
    
    return None

# Integration helper functions for the main app
def ensure_gcp_configured() -> bool:
    """Ensure GCP is configured, show setup if not"""
    
    auth_status = check_gcp_auth_status()
    
    if not auth_status["authenticated"]:
        st.warning("⚠️ Google Cloud Platform not configured")
        
        with st.expander("🔧 Configure Google Cloud Platform", expanded=True):
            auth_result = setup_gcp_authentication()
            return auth_result.get("authenticated", False)
    
    return True

def display_gcp_quick_setup():
    """Display a quick GCP setup component"""
    
    st.markdown("### 🌐 Quick GCP Setup")
    
    with st.container():
        auth_status = check_gcp_auth_status()
        
        if auth_status["authenticated"]:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.success(f"✅ Connected to: **{auth_status['project_id']}**")
            
            with col2:
                if st.button("🔄 Change", key="gcp_change"):
                    st.session_state.show_gcp_setup = True
            
            with col3:
                if st.button("🗑️ Clear", key="gcp_clear"):
                    clear_gcp_auth()
                    st.rerun()
        else:
            st.info("👆 Click to configure Google Cloud Platform")
            if st.button("🔧 Configure GCP", key="gcp_configure"):
                st.session_state.show_gcp_setup = True
        
        # Show setup if requested
        if st.session_state.get('show_gcp_setup', False):
            setup_gcp_authentication()
            
            if st.button("✅ Done", key="gcp_done"):
                st.session_state.show_gcp_setup = False
                st.rerun()

if __name__ == "__main__":
    # Test the component
    st.set_page_config(page_title="GCP Auth Test", layout="wide")
    st.title("🔐 Google Cloud Authentication Test")
    
    display_gcp_auth_setup()
