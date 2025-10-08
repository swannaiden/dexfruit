#!/usr/bin/env python3
# Script to automatically detect SpaceMouse and set permissions

import hid
import subprocess
import os

def list_devices_and_set_permissions(spacemouse_identifiers=None):
    """
    List all HID devices and set permissions for SpaceMouse devices.
    
    :param spacemouse_identifiers: List of tuples containing (vendor_id, product_id) or 
                                  strings that should appear in the product name for SpaceMouse devices
    """
    if spacemouse_identifiers is None:
        # Default identifiers for SpaceMouse devices
        # This is an example - you'll need to update with your actual SpaceMouse identifiers
        spacemouse_identifiers = [
            # Common 3Dconnexion VID (0x256f)
            (0x256f, None),  # Any 3Dconnexion product
            # You can add specific product IDs if needed
            # Or search by name
            "spacemouse", "space mouse", "3dconnexion"
        ]
    
    devices = hid.enumerate()
    
    if not devices:
        print("No HID devices found.")
        return
    
    spacemouse_paths = set()
    
    for i, device in enumerate(devices, start=1):
        vid = device['vendor_id']
        pid = device['product_id']
        product_name = device.get('product_string', '').lower()
        path = device['path'].decode() if isinstance(device['path'], bytes) else device['path']
        
        # Print device info
        print(f"\nDevice {i}:")
        print(f"  Vendor ID      : {hex(vid)}")
        print(f"  Product ID     : {hex(pid)}")
        print(f"  Manufacturer   : {device.get('manufacturer_string', 'Unknown')}")
        print(f"  Product        : {device.get('product_string', 'Unknown')}")
        print(f"  Serial Number  : {device.get('serial_number', 'Unknown')}")
        print(f"  Usage Page     : {hex(device['usage_page'])}")
        print(f"  Usage          : {hex(device['usage'])}")
        print(f"  Path           : {path}")
        print(f"  Interface      : {device.get('interface_number', 'Unknown')}")
        
        # Check if this device matches the SpaceMouse Compact identifier
        is_spacemouse = False
        product_string = device.get('product_string', '')
        
        for identifier in spacemouse_identifiers:
            if isinstance(identifier, tuple):
                vid_match, pid_match = identifier
                if vid_match is not None and vid == vid_match:
                    if pid_match is None or pid == pid_match:
                        is_spacemouse = True
                        break
            elif isinstance(identifier, str) and identifier in product_string:
                is_spacemouse = True
                print(f"  >>> Match found: '{identifier}' in '{product_string}'")
                break
        
        if is_spacemouse:
            print(f"  >>> SpaceMouse detected!")
            # Extract the /dev/hidraw* path
            if path.startswith('/dev/hidraw'):
                spacemouse_paths.add(path)
    
    # Set permissions for all found SpaceMouse devices
    if spacemouse_paths:
        print("\nSetting permissions for SpaceMouse devices:")
        for path in spacemouse_paths:
            set_permissions(path)
    else:
        print("\nNo SpaceMouse devices found.")

def set_permissions(path):
    """Set read/write permissions for the specified device path."""
    if not os.path.exists(path):
        print(f"  Error: {path} does not exist")
        return
        
    try:
        # Check current permissions
        current_perms = subprocess.check_output(['ls', '-l', path]).decode().strip()
        print(f"  Current permissions: {current_perms}")
        
        # Set new permissions
        print(f"  Setting permissions for: {path}")
        result = subprocess.run(['sudo', 'chmod', '666', path], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ Successfully set permissions for {path}")
            # Verify new permissions
            new_perms = subprocess.check_output(['ls', '-l', path]).decode().strip()
            print(f"  New permissions: {new_perms}")
        else:
            print(f"  ✗ Failed to set permissions: {result.stderr}")
    except Exception as e:
        print(f"  ✗ Error setting permissions: {str(e)}")

if __name__ == "__main__":
    # Specific identifier for SpaceMouse Compact
    custom_identifiers = [
        "SpaceMouse Compact"  # Exact product name string
    ]
    
    # Pass your custom identifiers or uncomment to use defaults
    list_devices_and_set_permissions(custom_identifiers)
    # list_devices_and_set_permissions()