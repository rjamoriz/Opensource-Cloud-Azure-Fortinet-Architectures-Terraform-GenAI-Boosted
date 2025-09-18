# Deployment of FortiGate-VM HA on the Google Cloud Platform (GCP) with 3 ports in single zone
## Introduction
A Terraform script to deploy FortiGate-VM HA (A-P) on GCP with 3 ports in single zone

## Requirements
* [Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html) >= 1.0.9
* Terraform Provider for Google Cloud Platform 3.90.0
* Terraform Provider for Google Cloud Platform Beta 3.90.0
* Terraform Provider for random 3.1.0
* A [GCP OAuth2 access token](https://developers.google.com/identity/protocols/OAuth2)

## Deployment overview
Terraform deploys the following components:
   - A Virtual Private Cloud (VPC) with one public subnet
   - A VPC with two private subnets
   - Two FortiGate-VM instances with three NICs
   - Three firewall rules: one for external, one for internal, one for sync/HA management.
   - One internal route associate to the private subnet
   - One forwarding rule with two different target groups.  Each target group points to the FortiGates.
   - During the HA failover, cluster ip, internal route, and forwarding rule will fail from the active unit to the passive unit.

## Deployment
To deploy the FortiGate-VM to GCP:
1. Clone the repository.
2. Obtain a GCP OAuth2 token and input it in the vars.tf file.
3. Customize variables in the `vars.tf` file as needed.
4. Initialize the providers and modules:
   ```sh
   $ cd XXXXX
   $ terraform init
    ```
5. Submit the Terraform plan:
   ```sh
   $ terraform plan
   ```
6. Verify output.
7. Confirm and apply the plan:
   ```sh
   $ terraform apply
   ```
8. If output is satisfactory, type `yes`.

Output will include the information necessary to log in to the FortiGate-VM instances:
```sh
FortiGate-HA-Active-MGMT-IP = XXX.XXX.XXX.XXX
FortiGate-HA-Cluster-IP = XXX.XXX.XXX.XXX
FortiGate-HA-Passive-MGMT-IP = XXX.XXX.XXX.XXX
FortiGate-Password = <password here>
FortiGate-Username = admin
Fowarding-IP-Address = "XXX.XXX.XXX.XXX"
```
*After deployment, user can add extra forwarding rule if needs to have extra public ip forwarding traffic to FortiGate-VM instances. User would need to configure the VIP to use with the forwarding rule.*


## Destroy the instance
To destroy the instance, use the command:
```sh
$ terraform destroy
```

# Support
Fortinet-provided scripts in this and other GitHub projects do not fall under the regular Fortinet technical support scope and are not supported by FortiCare Support Services.
For direct issues, please refer to the [Issues](https://github.com/fortinet/fortigate-terraform-deploy/issues) tab of this GitHub project.
For other questions related to this project, contact [github@fortinet.com](mailto:github@fortinet.com).

## License
[License](https://github.com/fortinet/fortigate-terraform-deploy/blob/master/LICENSE) © Fortinet Technologies. All rights reserved.
