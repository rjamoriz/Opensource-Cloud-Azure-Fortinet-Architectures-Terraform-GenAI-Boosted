resource "oci_core_volume" "vm_volume" {
  availability_domain = lookup(data.oci_identity_availability_domains.ads.availability_domains[var.availability_domain - 1], "name")
  compartment_id      = var.compartment_ocid
  display_name        = "fgt001-vm_volume"
  size_in_gbs         = var.volume_size
}
// Use paravirtualized attachment for now.
resource "oci_core_volume_attachment" "vm_volume_attach" {
  // attachment_type = "paravirtualized"
  attachment_type = "iscsi" //  user needs to manually add the iscsi disk on fos after
  compartment_id  = var.compartment_ocid
  instance_id     = oci_core_instance.vm.id
  volume_id       = oci_core_volume.vm_volume.id
}
