#!/bin/bash

# Google Cloud Platform Integration Setup Script
# FortiGate Multi-Cloud Deployment Assistant

echo "🌐 Setting up Google Cloud Platform Integration..."
echo "=================================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected"
    echo "   It's recommended to run this in a virtual environment"
    read -p "Continue anyway? (y/N): " continue_setup
    if [[ $continue_setup != "y" && $continue_setup != "Y" ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

# Install Google Cloud SDK dependencies
echo ""
echo "📦 Installing Google Cloud SDK dependencies..."
pip install --upgrade pip

# Install core GCP packages
echo "Installing core Google Cloud packages..."
pip install google-cloud-compute>=1.19.0
pip install google-cloud-resource-manager>=1.12.0
pip install google-cloud-iam>=2.15.0
pip install google-auth>=2.30.0
pip install google-auth-oauthlib>=1.2.0

# Install AI/ML packages
echo "Installing AI/ML packages..."
pip install google-cloud-aiplatform>=1.60.0
pip install google-cloud-speech>=2.25.0
pip install google-cloud-texttospeech>=2.18.0
pip install google-cloud-translate>=3.15.0

# Install monitoring and logging
echo "Installing monitoring and logging packages..."
pip install google-cloud-monitoring>=2.21.0
pip install google-cloud-logging>=3.11.0
pip install google-cloud-error-reporting>=1.13.0

# Install storage and additional utilities
echo "Installing storage and utility packages..."
pip install google-cloud-storage>=2.18.0
pip install google-cloud-bigquery>=3.25.0
pip install google-api-python-client>=2.140.0

# Install Terraform integration
echo "Installing Terraform integration..."
pip install python-terraform>=0.10.1

# Check if gcloud CLI is installed
echo ""
echo "🔧 Checking Google Cloud CLI..."
if command -v gcloud &> /dev/null; then
    echo "✅ Google Cloud CLI found"
    gcloud version
    
    # Check if user is authenticated
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "."; then
        echo "✅ Google Cloud authentication found"
        echo "Current account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
    else
        echo "⚠️  No active Google Cloud authentication found"
        echo "Please run: gcloud auth login"
        echo "Or: gcloud auth application-default login"
    fi
else
    echo "❌ Google Cloud CLI not found"
    echo ""
    echo "Please install the Google Cloud CLI:"
    echo "macOS: brew install google-cloud-sdk"
    echo "Linux: curl https://sdk.cloud.google.com | bash"
    echo "Windows: Download from https://cloud.google.com/sdk/docs/install"
fi

# Check if terraform is installed
echo ""
echo "🔧 Checking Terraform..."
if command -v terraform &> /dev/null; then
    echo "✅ Terraform found"
    terraform version
else
    echo "❌ Terraform not found"
    echo ""
    echo "Please install Terraform:"
    echo "macOS: brew install terraform"
    echo "Linux: Download from https://www.terraform.io/downloads.html"
    echo "Windows: Download from https://www.terraform.io/downloads.html"
fi

# Create GCP configuration directory
echo ""
echo "📁 Setting up configuration directories..."
mkdir -p ~/.fortigate-chatbot/gcp
mkdir -p ~/.fortigate-chatbot/gcp/credentials
mkdir -p ~/.fortigate-chatbot/gcp/terraform-state

echo "✅ Configuration directories created"

# Create environment variables template
echo ""
echo "📝 Creating environment variables template..."
cat > ~/.fortigate-chatbot/gcp/env_template.sh << 'EOF'
#!/bin/bash
# Google Cloud Platform Environment Variables
# Copy this file to env.sh and fill in your values

# Required: Your GCP Project ID
export GCP_PROJECT_ID="your-project-id"

# Optional: Service Account Credentials Path
# If not set, will use default application credentials
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Optional: Default Region and Zone
export GCP_DEFAULT_REGION="us-central1"
export GCP_DEFAULT_ZONE="us-central1-a"

# Optional: Terraform Backend Configuration
export GCP_TERRAFORM_BUCKET="your-terraform-state-bucket"

# Optional: Enable specific features
export ENABLE_VERTEX_AI="true"
export ENABLE_SPEECH_SERVICES="true"
export ENABLE_TRANSLATION="true"

echo "GCP environment variables loaded"
EOF

echo "✅ Environment template created at ~/.fortigate-chatbot/gcp/env_template.sh"

# Create service account setup script
echo ""
echo "📝 Creating service account setup script..."
cat > ~/.fortigate-chatbot/gcp/setup_service_account.sh << 'EOF'
#!/bin/bash
# Service Account Setup Script

echo "Setting up FortiGate Chatbot Service Account..."

# Check if project ID is set
if [[ -z "$GCP_PROJECT_ID" ]]; then
    echo "Error: GCP_PROJECT_ID environment variable not set"
    echo "Please set your project ID: export GCP_PROJECT_ID='your-project-id'"
    exit 1
fi

# Service account details
SA_NAME="fortigate-chatbot"
SA_DISPLAY_NAME="FortiGate Chatbot Service Account"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Create service account
echo "Creating service account..."
gcloud iam service-accounts create $SA_NAME \
    --display-name="$SA_DISPLAY_NAME" \
    --description="Service account for FortiGate Multi-Cloud Chatbot" \
    --project=$GCP_PROJECT_ID

# Grant necessary roles
echo "Granting IAM roles..."
ROLES=(
    "roles/compute.admin"
    "roles/resourcemanager.projectViewer"
    "roles/iam.serviceAccountUser"
    "roles/storage.admin"
    "roles/monitoring.viewer"
    "roles/logging.viewer"
    "roles/aiplatform.user"
)

for role in "${ROLES[@]}"; do
    echo "Granting $role..."
    gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role"
done

# Create and download key
echo "Creating service account key..."
KEY_FILE="$HOME/.fortigate-chatbot/gcp/credentials/service-account.json"
gcloud iam service-accounts keys create "$KEY_FILE" \
    --iam-account="$SA_EMAIL" \
    --project=$GCP_PROJECT_ID

echo "✅ Service account setup complete!"
echo "Key file location: $KEY_FILE"
echo "Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS='$KEY_FILE'"
EOF

chmod +x ~/.fortigate-chatbot/gcp/setup_service_account.sh
echo "✅ Service account setup script created"

# Create Terraform backend setup script
echo ""
echo "📝 Creating Terraform backend setup script..."
cat > ~/.fortigate-chatbot/gcp/setup_terraform_backend.sh << 'EOF'
#!/bin/bash
# Terraform Backend Setup Script

echo "Setting up Terraform backend for GCP..."

# Check if project ID is set
if [[ -z "$GCP_PROJECT_ID" ]]; then
    echo "Error: GCP_PROJECT_ID environment variable not set"
    exit 1
fi

# Create bucket for Terraform state
BUCKET_NAME="${GCP_PROJECT_ID}-terraform-state"
echo "Creating Cloud Storage bucket: $BUCKET_NAME"

gsutil mb -p $GCP_PROJECT_ID gs://$BUCKET_NAME

# Enable versioning
gsutil versioning set on gs://$BUCKET_NAME

# Set lifecycle policy
cat > lifecycle.json << 'LIFECYCLE_EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "isLive": false
        }
      }
    ]
  }
}
LIFECYCLE_EOF

gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME
rm lifecycle.json

echo "✅ Terraform backend setup complete!"
echo "Bucket name: $BUCKET_NAME"
echo "Update your terraform backend configuration to use this bucket"
EOF

chmod +x ~/.fortigate-chatbot/gcp/setup_terraform_backend.sh
echo "✅ Terraform backend setup script created"

# Test imports
echo ""
echo "🧪 Testing Google Cloud package imports..."
python3 -c "
try:
    from google.cloud import compute_v1
    from google.cloud import resource_manager_v3
    from google.auth import default
    print('✅ Core Google Cloud packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')

try:
    from google.cloud import aiplatform
    from google.cloud import speech
    from google.cloud import texttospeech
    print('✅ AI/ML packages imported successfully')
except ImportError as e:
    print(f'⚠️  AI/ML packages import warning: {e}')

try:
    import terraform
    print('✅ Terraform package imported successfully')
except ImportError as e:
    print(f'⚠️  Terraform package import warning: {e}')
"

echo ""
echo "🎉 Google Cloud Platform integration setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Set up your GCP project ID:"
echo "   export GCP_PROJECT_ID='your-project-id'"
echo ""
echo "2. Authenticate with Google Cloud:"
echo "   gcloud auth login"
echo "   gcloud auth application-default login"
echo ""
echo "3. (Optional) Set up service account:"
echo "   ~/.fortigate-chatbot/gcp/setup_service_account.sh"
echo ""
echo "4. (Optional) Set up Terraform backend:"
echo "   ~/.fortigate-chatbot/gcp/setup_terraform_backend.sh"
echo ""
echo "5. Configure environment variables:"
echo "   cp ~/.fortigate-chatbot/gcp/env_template.sh ~/.fortigate-chatbot/gcp/env.sh"
echo "   # Edit env.sh with your values"
echo "   source ~/.fortigate-chatbot/gcp/env.sh"
echo ""
echo "6. Test the integration by running the FortiGate Chatbot app"
echo ""
echo "For help and documentation, see:"
echo "- GCP_INTEGRATION_PLAN.md"
echo "- Google Cloud Documentation: https://cloud.google.com/docs"
echo ""
echo "🌐 Happy multi-cloud deploying!"
