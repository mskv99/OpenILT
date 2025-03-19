import sys
sys.path.append(".")
import time

import cv2 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

'''
Key concepts:
    A. Optical Kernels
    Optical kernels represent the point spread function (PSF) of the lithography system.
    They model how light propagates through the optical system and interacts with the mask.
    Different kernels are used for focus and defocus conditions to account for variations in the lithography process
    
    B. Aerial Image
    The aerial image is the light intensity distribution on the wafer.
    It is computed by convolving the mask pattern with the optical kernel
    
    C. Printed Image
    The printed image is the resist pattern on the wafer after exposure and development.
    It is computed by applying a sigmoid function to the aerial image, simulating the resist's response to light.
    
    D. Edge Placement Error (EPE)
    EPE measures the deviation between the simulated printed image and the target pattern.
    It is used to evaluate the quality of the lithography process and identify areas where the mask pattern needs correction.
'''

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REALTYPE = torch.float32
COMPLEXTYPE = torch.complex64

class Kernel:
    def __init__(self, basedir="./kernel", defocus=False, conjuncture=False, combo=False, device=DEVICE):
        '''
        Manages optical kernels used for lithography simulation

        How it works:
            - loads kernels from files, (e.g. focus.pt, defocus.pt)
            - supports different kernel configurations (e.g. focus, defocus, combo)
            - provides access to kernel data and scaling factors
        '''
        self._basedir = basedir
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self): 
        return self._kernels
        
    @property
    def scales(self): 
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus" + filename
        else:
            filename = "focus" + filename
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        filename = self._basedir + "/kernels/" + filename + ".pt"
        return filename

    def _scale_file(self):
        filename = self._basedir + "/scales/"
        if self._combo:
            return filename + "combo.pt"
        else:
            if self._defocus:
                return filename + "defocus.pt"
            else:
                return filename + "focus.pt"

def _maskFloat(mask, dose):
    '''
    Converts a binary mask into a floating-point mask with a specified dose
    '''
    return (dose * mask).to(COMPLEXTYPE)

def _kernelMult(kernel, maskFFT, kernelNum):
    '''
    Performs frequency domain multiplication of the masks's Fourier transform
    with the optical kernel. This operation is used to compute the aerial image,
    which represents the light intensity distribution on the wafer.

    How it works:
        1. Frequency-Domain Multiplication:
            The Fourier transform of the mask is multiplied with the optical kernel in the frequency domain.
            This operation is equivalent to convolution in the spatial domain but is computationally more efficient.
        2. Quadrant Handling:
            The frequency domain is divided into four quadrants, and each quadrant is processed independently.
            This ensures that the multiplication is performed correctly for both positive and negative frequencies.
        3. Batch Support:
            The method supports both single masks and batches of masks.
            For batches, the multiplication is performed independently for each mask in the batch.

    Args:
        kernel: the optical kernel (point spread function) in the frequency domain. Shape: [kernelNum, knx, kny]
        maskFFT: Fourier transform of the mask. Shape: [batch_size, 1, imx, imy] or [1, imx, imy]
        kernelNum: the number of kernels to use

    Returns:
        the result of multiplying mask's Fourier transform with the kernel in the frequency domain.

        shape: [batch_size, kernelNum, imx, imy] or [kernelNum, imx, imy]

    :meta public:
    '''
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:] # kernel dimensions (e.g. 35x35)
    knxh, knyh = knx // 2, kny // 2 # half kernel dimension (e.g. 17x17)

    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device) # Ensure kernel is on same device as maskFFT
    if len(maskFFT.shape) == 3: # Single mask (no batch dimension)
        # initialize output tensor, additional dimension for kernelNum, [kernelNum, imx, imy]
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        # Multiply top-left quadrant
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        # Multiply top-right quadrant
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        # Multiply bottom-left quadrant
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        # Multiply bottom-right quadrant
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else: # Batch of masks
        assert len(maskFFT.shape) == 4, f"[_kernelMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        # Multiply top-left quadrant
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        # Multiply top-right quadrant
        output[:, :, :knxh+1, -knyh:]  = maskFFT[:, :, :knxh+1, -knyh:]  * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        # Multiply bottom-left quadrant
        output[:, :, -knxh:, :knyh+1]  = maskFFT[:, :, -knxh:, :knyh+1]  * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        # Multiply bottom-right quadrant
        output[:, :, -knxh:, -knyh:]   = maskFFT[:, :, -knxh:, -knyh:]   * kernel[None, :kernelNum, :knxh, :knyh]
    return output


def _shift(cmask):
    shifted = torch.zeros_like(cmask)
    if len(shifted.shape) == 3: 
        shifted[:, :cmask.shape[-2]//2, :cmask.shape[-1]//2] = cmask[:, cmask.shape[-2]//2:, cmask.shape[-1]//2:]  # 1 = 4
        shifted[:, :cmask.shape[-2]//2, cmask.shape[-1]//2:] = cmask[:, cmask.shape[-2]//2:, :cmask.shape[-1]//2]  # 2 = 3
        shifted[:, cmask.shape[-2]//2:, :cmask.shape[-1]//2] = cmask[:, :cmask.shape[-2]//2, cmask.shape[-1]//2:]  # 3 = 2
        shifted[:, cmask.shape[-2]//2:, cmask.shape[-1]//2:] = cmask[:, :cmask.shape[-2]//2, :cmask.shape[-1]//2]  # 4 = 1
    else: 
        assert len(shifted.shape) == 4
        shifted[:, :, :cmask.shape[-2]//2, :cmask.shape[-1]//2] = cmask[:, :, cmask.shape[-2]//2:, cmask.shape[-1]//2:]  # 1 = 4
        shifted[:, :, :cmask.shape[-2]//2, cmask.shape[-1]//2:] = cmask[:, :, cmask.shape[-2]//2:, :cmask.shape[-1]//2]  # 2 = 3
        shifted[:, :, cmask.shape[-2]//2:, :cmask.shape[-1]//2] = cmask[:, :, :cmask.shape[-2]//2, cmask.shape[-1]//2:]  # 3 = 2
        shifted[:, :, cmask.shape[-2]//2:, cmask.shape[-1]//2:] = cmask[:, :, :cmask.shape[-2]//2, :cmask.shape[-1]//2]  # 4 = 1
    return shifted
def _centerMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    imx, imy = maskFFT.shape[-2:]
    knx, kny = kernel.shape[-2:]
    imxh, imyh = imx // 2, imy // 2
    knxh, knyh = knx // 2, kny // 2
    xstart = imxh - knx // 2
    ystart = imyh - kny // 2
    xend =  xstart + knx
    yend = ystart + kny
    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device)
    if len(maskFFT.shape) == 3: 
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, xstart:xend, ystart:yend] = maskFFT[:, xstart:xend, ystart:yend] * kernel[:kernelNum, :, :]
    else: 
        assert len(maskFFT.shape) == 4, f"[_centerMult]: Invalid shape of maskFFT: {maskFFT.shape}"
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, xstart:xend, ystart:yend] = maskFFT[:, :, xstart:xend, ystart:yend] * kernel[None, :kernelNum, :, :]
    return output


def _computeImageMatrix(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    assert len(cmask.shape) in [3, 4], f"[_computeImageMask]: Invalid shape: {cmask.shape}"
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp
def _computeImageMask(cmask, kernel, scale, kernelNum):
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    '''
    Computes the aerial image by applying the optical kernel to the mask in the frequency domain.
    Converts the mask into the frequency domain using a Fast Fourier Transform (FFT),
    multiplies it with the kernel, and then converts it back to the spatial domain
    using an Inverse FFT (IFFT).

    Args:
        cmask: The mask image. Shape: [batch_size, height, width] or [height, width], f.e. [16, 256, 256]
        kernel: The optical kernel (point spread function) in the frequency domain. Shape: [24, 35, 35]
        scale: Scaling factors for the kernels. Shape: [kernelNum], f.e. [24]
        kernelNum: the number of optical kernels to use. Shape: [], single integer number, f.e. {24}

    Returns:
        The aerial image in spatial dimension. Shape: [batch_size, kernelNum, height, width]
    '''
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    cmask = torch.unsqueeze(cmask, len(cmask.shape) - 2) # cmask: [16,1,256,256]
    cmask_fft = torch.fft.fft2(cmask, norm="forward") # cmask_fft: [16,1,256,256]
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)# tmp: [16, 24, 256, 256]
    tmp = torch.fft.ifft2(tmp, norm="forward")
    return tmp

def _convMatrix(cmask, dose, kernel, scale, kernelNum): 
    image = _computeImageMatrix(cmask, kernel, scale, kernelNum)
    return image
def _convMask(mask, dose, kernel, scale, kernelNum):
    '''
    Applies a dose to the mask and computes the aerial image using _computeImageMask.
    By multiplying on dose we convert binary mask (0 or 1) into floating point mask (0 or dose)

    Arguments:
        - mask: The mask image. Shape: [batch_size, height, width] or [height, width].
        - dose: The dose to apply to the mask into a floating point mask
        - kernel: The optical kernel. Shape: [kernelNum, kernel_height, kernel_width].
        - scale: Scaling factors for the kernels. Shape: [kernelNum].
        - kernelNum: The number of kernels to use

    Returns:
        The aerial image. Shape: [batch_size, kernelNum, height, width]

    '''
    cmask = _maskFloat(mask, dose) # cmask: [16, 256, 256]
    image = _computeImageMask(cmask, kernel, scale, kernelNum) # image: [16, 24, 256, 256]
    return image

def lithosim(mask, dose, kernel, scale, kernelNum): 
    '''
    Simulates the lithography process by computing the printed image from the aerial image.
    Applies the dose, scales the result, and sums over the kernels.

    Arguments:
        - mask: The mask image. Shape: [batch_size, height, width] or [height, width], f.e. [16, 256, 256]
        - dose: The dose to apply to the mask.
        - kernel: The optical kernel. Shape: [kernelNum, kernel_height, kernel_width]
        - scale: Scaling factors for the kernels. Shape: [kernelNum]
        - kernelNum: The number of kernels to use

    Returns:
        Square magnitude of tmp, sclaled by user defined value,
        summed over all kernels.

        Shape: [height, width] (single image) or [batch_size, height, width] (batch of images).
    '''
    # computing an aerial image
    tmp = _convMask(mask, dose, kernel, scale, kernelNum) # tmp: [16,24, 256, 256]
    if len(mask.shape) == 2:
        # reshaping scale for broadcasting
        scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2) # scale: [24,1,1]
        # computing the square magnitude of tmp, scaling it by scale, and summing over all kernels
        return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=0)
    else: 
        assert len(mask.shape) == 3, f"[_LithoSim.forward]: Invalid shape: {mask.shape}"
        scale = scale[:kernelNum].unsqueeze(0).unsqueeze(2).unsqueeze(3) # scale: [1, 24, 1, 1]
        return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=1)

def parseConfig(filename):
    '''
    Reads a configuration file and parses it into a dictionary
    The configuration file contains key-value pairs that define the parameters for the lithography
    simulation
    '''
    with open(filename, "r") as fin: 
        lines = fin.readlines()
    results = {}
    for line in lines: 
        splited = line.strip().split()
        if len(splited) >= 2: 
            key = splited[0]
            value = splited[1]
            results[key] = value
    return results
class LithoSim(nn.Module): # Mask -> Aerial -> Printed
    '''
    The LithoSim class is a PyTorch module that simulates the lithography process, transforming a mask
    into an aerial image and then into a printed image.

    Simulates the lithography process by:
        - Padding and scaling the input mask.
        - Computing the aerial image using optical kernels.
        - Converting the aerial image into a printed image using a sigmoid function.

    Config parameters:
        - KernelDir: Directory containing the optical kernels.
        - KernelNum: Number of kernels to use.
        - TargetDensity: Target intensity for the printed image.
        - PrintThresh: Threshold for the printed image.
        - PrintSteepness: Steepness of the sigmoid function.
        - DoseMax: Maximum dose.
        - DoseMin: Minimum dose.
        - DoseNom: Nominal dose.
        - Canvas: Size of the canvas (padded mask).
        - Resolution: Simulation resolution.
    '''
    def __init__(self, config="./config/lithoiccad13.txt"): 
        super(LithoSim, self).__init__()
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = parseConfig(config)
        required = ["KernelDir", "KernelNum", "TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in required: 
            assert key in self._config, f"[LithoSim]: Cannot find the config {key}."
        intfields = ["KernelNum", "Canvas", "Resolution", ]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
        # Read the kernels
        self._kernels = {"focus": Kernel(self._config["KernelDir"]), 
                         "defocus": Kernel(self._config["KernelDir"], defocus=True),}
        # Set the size
        self._canvas = self._config["Canvas"]
        self._res = self._config["Resolution"]

    def pad(self, mask):
        '''
        Pads the input mask to match the canvas size.
        If the mask is smaller than the canvas, it is padded with zeros to
        match the canvas size. The padding is centerer around the mask
        '''
        if mask.shape[-2] == self._canvas and mask.shape[-1] == self._canvas: 
            return mask
        result = None
        diffX = self._canvas - mask.shape[-2]
        diffY = self._canvas - mask.shape[-1]
        offsetX = round(diffX / 2)
        offsetY = round(diffY / 2)
        if len(mask.shape) == 2: 
            result = torch.zeros([self._canvas, self._canvas], dtype=REALTYPE, device=DEVICE)
            result[offsetX:offsetX+mask.shape[-2], offsetY:offsetY+mask.shape[-1]] = mask
        else: 
            assert len(mask.shape) == 3
            result = torch.zeros([mask.shape[0], self._canvas, self._canvas], dtype=REALTYPE, device=DEVICE)
            result[:, offsetX:offsetX+mask.shape[-2], offsetY:offsetY+mask.shape[-1]] = mask

        return result

    def unpad(self, image, mask):
        '''
        Removes padding to restore the original mask size.
        '''
        if mask.shape[-2] == self._canvas and mask.shape[-1] == self._canvas: 
            return mask
        result = None
        diffX = self._canvas - mask.shape[-2]
        diffY = self._canvas - mask.shape[-1]
        offsetX = round(diffX / 2)
        offsetY = round(diffY / 2)
        if len(mask.shape) == 2: 
            result = image[offsetX:offsetX+mask.shape[-2], offsetY:offsetY+mask.shape[-1]]
        else: 
            assert len(mask.shape) == 3
            result = image[:, offsetX:offsetX+mask.shape[-2], offsetY:offsetY+mask.shape[-1]] 

        return result
    
    def scaleForward(self, mask):
        '''
        Scales the mask down to the simulation resolution using bilinear interpolation.
        If the mask is already at the correct resolution, no scaling is performed
        '''
        if mask.shape[-2] == self._res and mask.shape[-1] == self._res: 
            return mask
        result = None
        if len(mask.shape) == 2: 
            result = F.interpolate(mask[None, None, :, :], size=(self._res, self._res))[0, 0]
        else: 
            assert len(mask.shape) == 3
            result = F.interpolate(mask[None, :, :, :], size=(self._res, self._res))[0]
        
        return result
    
    def scaleBackward(self, mask):
        '''
         Scales the printed image back to canvas size using bilinear interpolation.
         If the image is already at the correct size, no scaling is performed
        '''
        if mask.shape[-2] == self._canvas and mask.shape[-1] == self._canvas: 
            return mask
        result = None
        if len(mask.shape) == 2: 
            result = F.interpolate(mask[None, None, :, :], size=(self._canvas, self._canvas))[0, 0]
        else: 
            assert len(mask.shape) == 3
            result = F.interpolate(mask[None, :, :, :], size=(self._canvas, self._canvas))[0]
        
        return result
    
    def forward(self, mask):
        '''
        How it works:
            1. The mask is padded to match the canvas size and scaled to simulation resolution
            2. Aerial image is computed for nominal, maximum and minimum dose conditions.
            3. The aerial image is converted into a printed image using sigmoid function
            4. The printed image is scaled back to canvas size and unpadded (remove padding) to match the original mask size

        Returns:
            printed images for nominal, minimum and maximum dose conditions
        '''

        padded = self.pad(mask)
        scaled = self.scaleForward(padded)

        aerialNom = lithosim(scaled, self._config["DoseNom"], 
                             self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"])
        aerialMax = lithosim(scaled, self._config["DoseMax"], 
                             self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"])
        aerialMin = lithosim(scaled, self._config["DoseMin"], 
                             self._kernels["defocus"].kernels, self._kernels["defocus"].scales, self._config["KernelNum"])
        printedNom = torch.sigmoid(self._config["PrintSteepness"] * (aerialNom - self._config["TargetDensity"]))
        printedMax = torch.sigmoid(self._config["PrintSteepness"] * (aerialMax - self._config["TargetDensity"]))
        printedMin = torch.sigmoid(self._config["PrintSteepness"] * (aerialMin - self._config["TargetDensity"]))

        printedNom = self.scaleBackward(printedNom)
        printedMax = self.scaleBackward(printedMax)
        printedMin = self.scaleBackward(printedMin)

        printedNom = self.unpad(printedNom, mask)
        printedMax = self.unpad(printedMax, mask)
        printedMin = self.unpad(printedMin, mask)

        return printedNom, printedMax, printedMin


class PatchSim: 
    def __init__(self, simulator, sizeX, sizeY, scale=1): # sizeY is the dim 0
        self._simulator = simulator
        self._sizeX = sizeX
        self._sizeY = sizeY
        self._scale = scale

    def getSize(self, coords): 
        max0 = 0
        max1 = 0
        for idx, coord in enumerate(coords): 
            coord0 = round(self._scale*(coord[1]+self._sizeY))
            coord1 = round(self._scale*(coord[0]+self._sizeX))
            if coord0 > max0: 
                max0 = coord0
            if coord1 > max1: 
                max1 = coord1
        return coord0, coord1
    
    def concat(self, patches, coords):
        # compute the size of the full-chip image, height and width
        max0, max1 = self.getSize(coords)
        # initialize 2 tensors:
        # concated - a tensor to store the combined full-chip image, initialized with zeros
        # counts - a tensor to track the number of patches contributing to each pixel in the full-chip image
        concated = torch.zeros([max0, max1], dtype=REALTYPE, device=DEVICE)
        counts = torch.zeros([max0, max1], dtype=REALTYPE, device=DEVICE)
        for idx, coord in enumerate(coords): # iterate over each patch and its bottom-left coordinates
            coord0a = round(self._scale*coord[1]) # start row in the full-chip image, coord0a: 0
            coord1a = round(self._scale*coord[0]) # start column in the full-chip image, coord1a:0
            coord0b = min(coord0a + patches[idx].shape[0], concated.shape[0]) # end row, coord0b: 250
            coord1b = min(coord1a + patches[idx].shape[1], concated.shape[1]) # end column, coord0b:250
            # a mask is created to exclude boundary regions from the patch
            # the mask is set to 1 in the valid region and 0 in the boundary region
            # the valid region is defined as the central 50% of the patch (25% from each patch is excluded)
            masked = torch.zeros_like(patches[idx]) # masked: (250, 250)
            valid0 = round(patches[idx].shape[0] * 0.25) # valid region height: 62.5
            valid1 = round(patches[idx].shape[1] * 0.25) # valid region width: 62.5
            if coord0a == 0 and coord1a == 0: 
                masked[:, :] = 1 # include the entire patch
            elif coord0a == 0: 
                masked[:, valid1:-valid1] = 1 # exclude left and right boundaries
            elif coord1a == 0: 
                masked[valid0:-valid0, :] = 1 # exclude top and bottim boundaries
            else: 
                masked[valid0:-valid0, valid1:-valid1] = 1 # exclude left/right and top/bottom boundaries
            coord0a = max(coord0a, 0)
            coord1a = max(coord1a, 0)
            start0 = max(-coord0a, 0)
            start1 = max(-coord1a, 0)
            size0 = coord0b - coord0a
            size1 = coord1b - coord1a
            # print(concated.shape, patches[idx].shape, masked.shape, coord0a, coord0b, start0, size0, coord1a, coord1b, start1, size1)
            concated[coord0a:coord0b, coord1a:coord1b] += (patches[idx] * masked)[start0:size0, start1:size1]
            counts[coord0a:coord0b, coord1a:coord1b] += (torch.ones_like(patches[idx]) * masked)[:size0, :size1]
        counts[counts <= 1e-3] = 1e-3 # avoiding division by zero
        concated = concated / counts # normalizing
        return concated
    
    def simulate(self, crops, coords, batchsize=16): 
        savedMask = []
        savedNom = []
        savedMax = []
        savedMin = []
        for idx in tqdm(range(0, len(crops), batchsize)): 
            batch = []
            for jdx in range(idx, min(len(crops), idx+batchsize)): 
                crop = crops[jdx]
                image = poly2img(crop, sizeX=self._sizeX, sizeY=self._sizeY, scale=self._scale) / 255
                image = torch.tensor(image, dtype=REALTYPE, device=DEVICE) # image: (250, 250)
                batch.append(image)
            image = torch.stack(batch, dim=0) # batch: (2, 250, 250)
            pNom, pMax, pMin = self._simulator(image) # pNom, pMax, pMin: (2, 250, 250)
            for jdx in range(idx, min(len(crops), idx+batchsize)): 
                index = jdx - idx
                savedMask.append(image[index])
                savedNom.append(pNom[index])
                savedMax.append(pMax[index])
                savedMin.append(pMin[index])

        # for crop in tqdm(crops): 
        #     image = poly.poly2img(crop, sizeX=self._sizeX, sizeY=self._sizeY, scale=self._scale) / 255
        #     image = torch.tensor(image, dtype=REALTYPE, device=DEVICE)
        #     pNom, pMax, pMin = self._simulator(image)
        #     savedMask.append(image)
        #     savedNom.append(pNom)
        #     savedMax.append(pMax)
        #     savedMin.append(pMin)

        origin = self.concat(savedMask, coords)
        bignom = self.concat(savedNom, coords)
        bigmax = self.concat(savedMax, coords)
        bigmin = self.concat(savedMin, coords)

        return bignom, bigmax, bigmin, origin
    
    def checkEPE(self, segments, bignom, origin, distance=16, details=False): 
        bignom = bignom.detach().cpu().numpy()
        bignom = (bignom > 0.499)
        origin = origin.detach().cpu().numpy()
        origin = (origin > 0.499)

        begins = list(map(lambda x: x[0], segments))
        ends = list(map(lambda x: x[1], segments))
        begins = np.array(begins, dtype=np.int32)
        ends = np.array(ends, dtype=np.int32)
        mids = (begins + ends) / 2
        begins = np.round(self._scale*begins).astype(dtype=np.int32)
        ends = np.round(self._scale*ends).astype(dtype=np.int32)
        mids = np.round(self._scale*mids).astype(dtype=np.int32)
        equals = (ends == begins)

        allequal = np.logical_and(equals[:, 0], equals[:, 1])
        verticals = np.logical_and(equals[:, 0], np.logical_not(allequal))
        horizontal = np.logical_and(equals[:, 1], np.logical_not(allequal))
        lefts = mids.copy()
        lefts[:, 0] += 2
        lefts = origin[lefts[:, 1], lefts[:, 0]]
        lefts = np.logical_and(verticals, lefts)
        rights = mids.copy()
        rights[:, 0] -= 2
        rights = origin[rights[:, 1], rights[:, 0]]
        rights = np.logical_and(verticals, rights)
        ups = mids.copy()
        ups[:, 1] -= 2
        ups = origin[ups[:, 1], ups[:, 0]]
        ups = np.logical_and(horizontal, ups)
        downs = mids.copy()
        downs[:, 1] += 2
        downs = origin[downs[:, 1], downs[:, 0]]
        downs = np.logical_and(horizontal, downs)
        lr = np.logical_and(lefts, rights)
        ud = np.logical_and(ups, downs)
        lefts = np.logical_and(lefts, np.logical_not(lr))
        rights = np.logical_and(rights, np.logical_not(lr))
        ups = np.logical_and(ups, np.logical_not(ud))
        downs = np.logical_and(downs, np.logical_not(ud))

        inners = mids.copy()
        inners[lefts, 0] += round(self._scale * distance)
        inners[rights, 0] -= round(self._scale * distance)
        inners[ups, 1] -= round(self._scale * distance)
        inners[downs, 1] += round(self._scale * distance)
        outers = mids.copy()
        outers[lefts, 0] -= round(self._scale * distance)
        outers[rights, 0] += round(self._scale * distance)
        outers[ups, 1] += round(self._scale * distance)
        outers[downs, 1] -= round(self._scale * distance)

        validsIn = origin[inners[:, 1], inners[:, 0]]
        validsOut = np.logical_not(origin[outers[:, 1], outers[:, 0]])
        viosIn = np.logical_not(bignom[inners[:, 1], inners[:, 0]])
        viosIn = np.logical_and(viosIn, validsIn)
        viosOut = bignom[outers[:, 1], outers[:, 0]]
        viosOut = np.logical_and(viosOut, validsOut)
        viosAll = np.logical_or(viosIn, viosOut)

        epe = np.sum(viosAll)
        print(f"EPE violations: {epe}={np.sum(viosIn)}+{np.sum(viosOut)}/{np.sum(validsIn)+np.sum(validsOut)}/{2*len(segments)}")

        if details: 
            hmoves = np.zeros((len(segments), ), dtype=np.int32)
            hmoves[np.logical_and(lefts, viosIn)] = -1
            hmoves[np.logical_and(rights, viosIn)] = +1
            hmoves[np.logical_and(lefts, viosOut)] = +1
            hmoves[np.logical_and(rights, viosOut)] = -1
            vmoves = np.zeros((len(segments), ), dtype=np.int32)
            vmoves[np.logical_and(ups, viosIn)] = +1
            vmoves[np.logical_and(downs, viosIn)] = -1
            vmoves[np.logical_and(ups, viosOut)] = -1
            vmoves[np.logical_and(downs, viosOut)] = +1
            return epe, viosAll, hmoves, vmoves

        return epe
    
    def validate(self, segments, origin): 
        origin = origin.detach().cpu().numpy()
        origin = (origin > 0.499)

        begins = list(map(lambda x: x[0], segments))
        ends = list(map(lambda x: x[1], segments))
        begins = np.array(begins, dtype=np.int32)
        ends = np.array(ends, dtype=np.int32)
        mids = (begins + ends) / 2
        begins = np.round(self._scale*begins).astype(dtype=np.int32)
        ends = np.round(self._scale*ends).astype(dtype=np.int32)
        mids = np.round(self._scale*mids).astype(dtype=np.int32)
        equals = (ends == begins)

        allequal = np.logical_and(equals[:, 0], equals[:, 1])
        verticals = np.logical_and(equals[:, 0], np.logical_not(allequal))
        horizontal = np.logical_and(equals[:, 1], np.logical_not(allequal))
        lefts = mids.copy()
        lefts[:, 0] += 2
        lefts = origin[lefts[:, 1], lefts[:, 0]]
        lefts = np.logical_and(verticals, lefts)
        rights = mids.copy()
        rights[:, 0] -= 2
        rights = origin[rights[:, 1], rights[:, 0]]
        rights = np.logical_and(verticals, rights)
        ups = mids.copy()
        ups[:, 1] -= 2
        ups = origin[ups[:, 1], ups[:, 0]]
        ups = np.logical_and(horizontal, ups)
        downs = mids.copy()
        downs[:, 1] += 2
        downs = origin[downs[:, 1], downs[:, 0]]
        downs = np.logical_and(horizontal, downs)
        lr = np.logical_and(lefts, rights)
        ud = np.logical_and(ups, downs)
        lefts = np.logical_and(lefts, np.logical_not(lr))
        rights = np.logical_and(rights, np.logical_not(lr))
        ups = np.logical_and(ups, np.logical_not(ud))
        downs = np.logical_and(downs, np.logical_not(ud))
        valids = np.logical_or(np.logical_or(lefts, rights), np.logical_or(ups, downs))

        print(f"Validated: {np.sum(valids)}/{len(valids)}")

        return valids, lefts, rights, ups, downs

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from utils.layout import getCrops, getShapes, readLayout
    from utils.polygon import dissect, segs2poly, poly2imgShifted, poly2img

    try:
        import pya
    except Exception:
        import klayout.db as pya


    ORIG_VERSION = False
    if ORIG_VERSION:
        infile = readLayout("benchmark/gcd_45nm.gds", 11)
        shapes, poses = getShapes(infile, layer=11, maxnum=None, verbose=False)
        segments = []
        polygons = []
        for datum, coord in zip(shapes, poses):
            polygon = list(map(lambda x: (x[0]+coord[0], x[1]+coord[1]), datum))
            dissected = dissect(polygon, lenCorner=35, lenUniform=70)
            reconstr = segs2poly(dissected)
            segments.extend(dissected)
            polygons.append(reconstr)
        print(f"In total {len(polygons)} shapes")

        crops, coords = getCrops(infile, layer=11, sizeX=1200, sizeY=1600, strideX=570, strideY=700, maxnum=None, verbose=False)
        print(f"In total {len(crops)} crops")

        bigsim = PatchSim(LithoSim(), sizeX=1200, sizeY=1600, scale=0.125)
        bignom, bigmax, bigmin, origin = bigsim.simulate(crops, coords)
        print(bignom.shape)

        epe = bigsim.checkEPE(segments, bignom, origin, distance=16)
        print(f"EPE violation: {epe}")

        origin = origin.detach().cpu().numpy()
        bignom = bignom.detach().cpu().numpy()
        cv2.imwrite("tmp/origin.png", np.flip(origin*255, axis=0))
        cv2.imwrite("tmp/simed.png", np.flip(bignom*255, axis=0))

        plt.subplot(1, 2, 1)
        plt.imshow(origin, origin="lower")
        plt.subplot(1, 2, 2)
        plt.imshow(bignom, origin="lower")
        plt.show()
    else:
        infile = readLayout(filename = 'tmp/test1.gds', layer = 0)
        crops, coords = getCrops(infile, layer=0, sizeX=2000, sizeY=2000, strideX=1000, strideY=1000, maxnum=2, verbose=True)
        # combined_crop, combined_coord = getCrops(infile, layer=0, sizeX=2000, sizeY=4000, strideX=2000, strideY=4000, maxnum=2, verbose=False)
        print(f"In total {len(crops)} crops")

        first_poly = list(list(map(lambda x: (x[0] + 2000, x[1] + 2000), crops[0][0])))
        plt.subplot(1, 2, 1)
        plt.imshow(poly2imgShifted(crops[0], sizeX=2000, sizeY=2000))
        plt.subplot(1, 2, 2)
        plt.imshow(poly2imgShifted(crops[1], sizeX=2000, sizeY=2000))
        # plt.subplot(1, 3, 3)
        # plt.imshow(poly2imgShifted(first_poly + crops[1], sizeX=2000, sizeY=4000))
        plt.show()

        bigsim = PatchSim(LithoSim(), sizeX=2000, sizeY=2000, scale=0.125)
        bignom, bigmax, bigmin, origin = bigsim.simulate(crops, coords)
        print(f'Bignom shape: {bignom.shape}')
        print(f'Origin shape: {origin.shape}')

        # plt.subplot(1, 2, 1)
        # plt.imshow(poly2imgShifted(origin.numpy(), sizeX=2000, sizeY=4000))
        # plt.subplot(1, 2, 2)
        # plt.imshow(poly2imgShifted(bignom.numpy(), sizeX=2000, sizeY=4000))
        # plt.imshow()

        # in case want to save an image with cv2
        cv2.imshow('Displaying concatenated patches from layout', np.flip(origin.detach().cpu().numpy()*255, axis=0))
        cv2.waitKey(0)

        cv2.imshow('Displaying lithography simulation on concatenated patches from layout', np.flip(bignom.detach().cpu().numpy()*255, axis=0))
        cv2.waitKey(0)



