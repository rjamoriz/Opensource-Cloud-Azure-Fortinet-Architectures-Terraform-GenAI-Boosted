import os
import subprocess

# Directory where Terraform templates are stored
def execute_terraform_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.decode('utf-8')}"

def initialize_terraform(directory: str) -> str:
    command = f"cd {directory} && terraform init"
    return execute_terraform_command(command)

def apply_terraform(directory: str) -> str:
    command = f"cd {directory} && terraform apply -auto-approve"
    return execute_terraform_command(command)

def destroy_terraform(directory: str) -> str:
    command = f"cd {directory} && terraform destroy -auto-approve"
    return execute_terraform_command(command)

def check_terraform_status(directory: str) -> str:
    command = f"cd {directory} && terraform plan"
    return execute_terraform_command(command)

def list_templates(base_directory: str) -> list:
    """
    List all subdirectories in the base directory that contain Terraform templates.
    """
    return [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
