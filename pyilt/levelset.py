import os
import sys
sys.path.append(".")
sys.path.append(os.path.dirname(__file__))
import time
import math
import multiprocessing as mp

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
# import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

class LevelSetCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBL2", "WeightPVBand", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBL2", "WeightPVBand", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]
    

def gradImage(image): 
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = func.pad(image, (1, 1, 1, 1), mode='replicate')[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)
    
class _Binarize(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, levelset): 
        ctx.save_for_backward(levelset)
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask
    
    @staticmethod
    def backward(ctx, grad_output): 
        levelset, = ctx.saved_tensors
        gradX, gradY = gradImage(levelset)
        l2norm = torch.sqrt(gradX**2 + gradY**2)
        return -l2norm * grad_output
    
class Binarize(nn.Module): 
    def __init__(self): 
        super(Binarize, self).__init__()
        pass

    def forward(self, levelset): 
        return _Binarize.apply(levelset)

class LevelSet(nn.Module): 
    def __init__(self, lithosim): 
        super(LevelSet, self).__init__()
        self._binarize = Binarize()
        self._lithosim = lithosim
        # self.add_module("binary", self._binarize)
        # self.add_module("lithosim", self._lithosim)

    def forward(self, params): 
        mask = self._binarize(params)
        printedNom, printedMax, printedMin = self._lithosim(mask)
        return mask, printedNom, printedMax, printedMin


class LevelSetILT: 
    def __init__(self, config=LevelSetCfg("./config/pylevelset2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(LevelSetILT, self).__init__()
        self._config = config
        self._device = device
        # LevelSet
        self._levelset = LevelSet(lithosim).to(DEVICE)
        if multigpu: 
            self._levelset = nn.DataParallel(self._levelset)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
    
    def solve(self, target, params, curv=None, verbose=0): 
        # Initialize
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        # opt = optim.SGD([params], lr=self._config["StepSize"])
        opt = optim.Adam([params], lr=self._config["StepSize"])
        
        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]): 
            mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter + backup * (1.0 - self._filter))
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask

def parallel(): 
    SCALE = 4
    l2s = []
    pvbs = []
    epes = []
    shots = []
    targetsAll = []
    paramsAll = []
    cfg   = LevelSetCfg("./config/pylevelset512.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = LevelSetILT(cfg, litho, multigpu=True)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11): 
        print(f"[PyLevelset]: Preparing testcase {idx}")
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.LevelSetInitTorch().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        targetsAll.append(torch.unsqueeze(target, 0))
        paramsAll.append(torch.unsqueeze(params, 0))
    count = torch.cuda.device_count()
    print(f"Using {count} GPUs")
    while count > 0 and len(targetsAll) % count != 0: 
        targetsAll.append(targetsAll[-1])
        paramsAll.append(paramsAll[-1])
    print(f"Augmented to {len(targetsAll)} samples. ")
    targetsAll = torch.cat(targetsAll, 0)
    paramsAll = torch.cat(paramsAll, 0)

    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(targetsAll, paramsAll)
    runtime = time.time() - begin

    for idx in range(1, 11): 
        mask = bestMask[idx-1]
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.LevelSetInitTorch().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(mask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(mask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = shotCount.run(mask, shape=(512, 512))
        cv2.imwrite(f"./tmp/LevelSet_test{idx}.png", (mask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.0f}; Shot {np.mean(shots):.0f}; SolveTime {runtime:.2f}s")

def serial(): 
    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    targetsAll = []
    paramsAll = []
    cfg   = LevelSetCfg("./config/pylevelset2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = LevelSetILT(cfg, litho)
    for idx in range(1, 11): 
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.LevelSetInitTorch().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None)
        runtime = time.time() - begin
        
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.LevelSetInitTorch().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)
        cv2.imwrite(f"./tmp/LevelSet_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        runtimes.append(runtime)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime: {np.mean(runtimes):.2f}s")

if __name__ == "__main__": 
    serial()
    # parallel()
