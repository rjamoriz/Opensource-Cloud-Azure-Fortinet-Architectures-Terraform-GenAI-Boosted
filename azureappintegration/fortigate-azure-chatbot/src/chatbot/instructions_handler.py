from typing import List

def format_instructions(instructions: List[str]) -> str:
    """Format the instructions for deployment."""
    formatted_instructions = "\n".join(f"{idx + 1}. {instruction}" for idx, instruction in enumerate(instructions))
    return formatted_instructions

def trigger_deployment_commands(deployment_type: str) -> str:
    """Trigger the appropriate deployment commands based on the deployment type."""
    if deployment_type.lower() == "byol":
        return (
            "BYOL az vm image terms accept --publisher fortinet "
            "--offer fortinet_fortigate-vm_v5 --plan fortinet_fg-vm"
        )
    elif deployment_type.lower() == "payg":
        return (
            "PAYG az vm image terms accept --publisher fortinet "
            "--offer fortinet_fortigate-vm_v5 --plan fortinet_fg-vm_payg_2023"
        )
    else:
        return "Invalid deployment type specified. Please choose 'BYOL' or 'PAYG'."

def process_instructions(instructions: List[str], deployment_type: str) -> str:
    """Process the instructions and trigger deployment commands."""
    formatted_instructions = format_instructions(instructions)
    deployment_command = trigger_deployment_commands(deployment_type)
    
    return f"Instructions:\n{formatted_instructions}\n\nDeployment Command:\n{deployment_command}"