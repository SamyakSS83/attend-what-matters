import numpy as np
import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.pixel_data_handlers.util import apply_voi_lut


def check_transfer_id(dicom_file):

    ### Function to check whether Transfer Syntax UID is present or not, if not present manually add a default value
    
    if hasattr(dicom_file.file_meta, 'TransferSyntaxUID'):
        return True
    else:
        return False


def convert_to_numpy(dicom_image):

    ### Function to convert a pydicom image to numpy pixel array (uint16)

    if(not check_transfer_id(dicom_image)):
        dicom_image.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    image_bits = dicom_image.BitsStored
    
    image_pixels = apply_voi_lut(dicom_image.pixel_array, dicom_image)
    if dicom_image.PhotometricInterpretation == "MONOCHROME2":
        image_pixels = image_pixels - np.min(image_pixels)
    elif dicom_image.PhotometricInterpretation == 'RGB':
        image_pixels = image_pixels - np.min(image_pixels, axis = (0, 1)).reshape(1, 1, -1)
    else:
        image_pixels = np.amax(image_pixels) - image_pixels
    image_pixels = image_pixels / np.max(image_pixels)
    image_pixels = (image_pixels * 65535).astype(np.uint16)

    return image_pixels if len(image_pixels.shape) == 2 else image_pixels[:, :, 0]


def convert_16bit_image_to_8bit(image):

    downsampled_image = ((image / 65535) * 255).astype(np.uint8)

    return downsampled_image

