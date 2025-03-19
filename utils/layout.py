import os
import sys
import math
import pickle
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union, Any
from utils.polygon import poly2img, poly2imgShifted

try: 
    import pya
except Exception: 
    import klayout.db as pya

def getCell(infile, layer, cell):
    """
    Extracts a specific cell from a layout(infile) and copies the shapes
    on a specified layer to a new layout

    How it works:
        - reads a cell by its name
        - flatten hierarchical structures to a single level
        - copy shapes from the given layer in infile to output layout

    Args:
        infile: input file with layout for processing
        layer: number of layer to operate with
        cell: name of the cell to operate with

    Returns:
        layout object
    """
    ly = infile
    cropped = ly.cell(cell)
    cropped.flatten(-1)
    layout = pya.Layout()
    layout.dbu = ly.dbu
    top = layout.create_cell(cell)
    layerFr = infile.layer(layer, 0)
    layerTo = layout.layer(layer, 0)
    for instance in cropped.each_shape(layerFr): 
        top.shapes(layerTo).insert(instance)
    return layout


def readLayout(filename:str, layer:int, crop:bool=True)->pya.Layout:
    """
    Load layout data and optionally crop the geometry to specific region

    How it works:
    - reads the layout file(filename) and extracts the bounding box (bbox) coordinates
    - crops the layout if crop=True, using cropLayout

    Args:
        - filename: path to layout file, (e.g. GDSII)
        - layer: the layer to extract from the layout
        - crop: whether to crop the layout to the bounding box of the top cell

    Returns:
        A cropped or uncropped layout object
    """
    infile = pya.Layout()
    infile.read(filename)
    bbox = infile.top_cell().bbox()
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    print(f"Read layout of geometry ({left, bottom})-({right, top})")
    if crop: 
        cropped = cropLayout(infile, layer, left, bottom, right, top)
    else: 
        cropped = infile
    return cropped

def createLayout(polygons:List[List[Tuple[int, int]]],
                 layer:int,
                 dbu:float)->pya.Layout:
    """
    Construct a fresh layout with user-defined polygons.
    Creates a new layout and top-level cell.
    Inserts each polygon as a shape into the specified layer

    How it works:
        - Each polygon is represented by a list of points.
        - These points are transformed into pya.Point objects
        - A new instance of pya.SimplePolygon is created from these points.
        - The polygon is inserted into the top cell of the layout on the specified layer

    Args:
        polygons: a list of polygons, where each polygon is a list of (x,y) points
        layer: the target layer for the polygons
        dbu: database unit (scaling factor for the layout)

    Returns:
        a layout object
    """
    layout = pya.Layout()
    layout.dbu = dbu
    top = layout.create_cell("TOP")
    layer = layout.layer(layer, 0)
    for polygon in polygons: 
        points = [pya.Point(point[0], point[1]) for point in polygon]
        instance = pya.SimplePolygon(points)
        top.shapes(layer).insert(instance)
    return layout

def cropLayout(infile:str,
               layer:int,
               beginX:int,
               beginY:int,
               endX:int,
               endY:int)->pya.Layout:
    """
    Extract specific portions of the layout for further processing

    How it works:
        - scales coordinates based on database units
        - clipping is performed using a bounding box (pya.Box)
        - flattens the structure and inserts merged shapes back into a new layout

    Args:
        infile: input layout object
        layer: the layer to extract
        beginX, beginY, endX, endY: coordinates of the cropping region in nm

    Returns:
        a cropped layout object
    """
    # 1. Scaling coordinates
    # The scale value converts the input coordinates (in nanometers) to layout database units (dbu).
    # This ensures that the coordinates provided by the user match the internal representation of the layout.
    ly = infile
    scale = 0.001 / ly.dbu
    beginX = round(beginX * scale)
    beginY = round(beginY * scale)
    endX = round(endX * scale)
    endY = round(endY * scale)
    print(f'beginX:{beginX}')
    print(f'beginY:{beginY}')
    print(f'endX:{endX}')
    print(f'endY:{endY}')

    # 2. Creating a cropping box
    # pya.Box is a geometric rectangle that defines the region of interest in the layout.
    cbox = pya.Box.new(beginX, beginY, endX, endY)
    # 3. Cropping with ly.clip()
    # This line performs the actual cropping operation
    #   - ly.top_cell().cell_index() gets the index of the top cell in the layout hierarchy.
    #   - cbox specifies the region to clip.
    cropped = ly.clip(ly.top_cell().cell_index(), cbox)
    # 4. Flattening the cropped cell
    # ly.cell(cropped) accesses the newly created cell.
    # flatten(-1) removes the hierarchical structure and brings all geometry into a single layer to simplify operations.
    cropped = ly.cell(cropped)
    cropped.flatten(-1)

    layout = pya.Layout()
    layout.dbu = ly.dbu
    top = layout.create_cell("TOP")
    layerFr = infile.layer(layer, 0)
    layerTo = layout.layer(layer, 0)
    # 5. Creating a pya.Region and merging shapes
    # pya.Region is a data structure that represents a collection of geometric shapes
    # It supports various geometric operations such as merging, boolean operations and area calculations
    # Merging combines overlapping or adjacent shapes into a single unified polygon. This helps simplify
    # the resulting layout reducing complexity of downstream operations
    region = pya.Region(cropped.begin_shapes_rec(layerFr))
    region.merge()
    # 6. After merging, the simplified shapes are inserted into the target layer of the new layout.
    top.shapes(layerTo).insert(region)
    # for instance in cropped.each_shape(layerFr): 
    #     top.shapes(layerTo).insert(instance)
    return layout


def shape2points(shape, verbose=False)->List[Tuple[Any, Any]]:
    """
    Transform layout shapes(box, path, polygon) into point-based data for geometric analysis.

    How it works:
        - detects shape types and converts them to lists of points
        - handles boxes, paths and polygons specifically

    Args:
        shape: the particular layot shape(box, path,polygon)
        verbose: whether to print debug information

    Note:
        Here point coordinates are in layout database units (DBU).
        To convert them back to real-world units use the following rule:

        .. parsed-literal::
            x_um = x_dbu * ly.dbu
            x_nm = x_dbu * ly.dbu * 1000

    Returns:
        list of points
    """
    points = []
    if shape.is_box(): 
        box = shape.box
        points.append((box.left, box.bottom))
        points.append((box.left, box.top))
        points.append((box.right, box.top))
        points.append((box.right, box.bottom))
        if verbose:
            print(f"Box: ({box.bottom}, {box.left}, {box.top}, {box.right})")
    elif shape.is_path(): 
        path = shape.path
        polygon = path.simple_polygon()
        for point in polygon.each_point(): 
            points.append((point.x, point.y))
        if verbose:
            print(f"Path: ({polygon})")
            for point in polygon.each_point(): 
                print(f" -> Point: {point}")
    elif shape.is_polygon(): 
        polygon = shape.polygon
        polygon = polygon.to_simple_polygon()
        for point in polygon.each_point(): 
            points.append((point.x, point.y))
        if verbose:
            print(f"Polygon: ({polygon})")
            for point in polygon.each_point(): 
                print(f" -> Point: {point.x}, {point.y}")
    valid = shape.is_box() or shape.is_path() or shape.is_polygon() or shape.is_text() or shape.is_edge()
    assert valid, f"ERROR: Invalid shape: {shape}"

    return points

def yieldShape(infile:pya.Layout, layer:int, verbose:bool=True): # unit: nm
    """
    Iterates over shapes in a layout and yields their points and
    bounding box coordinates

    How it works:
        - scales coordinates to match layout dbu
        - extracts shapes from specified layer
        - converts each shape to points using shape2points
        - yields the points and bounding box coordinates

    Args:
        infile: the input layout object
        layer: the layer to extract shapes from
        verbose: whether to print debug information

    Yields:
        (points, (minX, minY, maxX, maxY))

    Note:
        Subtracting minX and minY shifts the coordiantes so that the minimum corner of the
        bounding box becomes the origin (0,0). This is done to normalize coordinates and
        simplify further processing. By dividing on scale we convert layout coordinates
        to real-world units, DBU -> nm

    Example:
        .. parsed-literal::
            Polygon: ((57600,268600;57600,273850;58300,273850;58300,268600))
             -> Point: 57600, 268600
             -> Point: 57600, 273850
             -> Point: 58300, 273850
             -> Point: 58300, 268600
            N-th shape of the polygon: ([(0, 0), (0, 525), (70, 525), (70, 0)], (5760, 26860, 5830, 27385, 5795, 27122))

    """
    ly = infile
    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell()
    topcell.flatten(-1)
    layer = ly.layer(layer, 0)
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu

    shapes = topcell.shapes(layer)
    for shape in shapes.each(): 
        points = shape2points(shape, verbose=verbose)
        if len(points) > 0: 
            minX = min(map(lambda x: x[0], points))
            minY = min(map(lambda x: x[1], points))
            maxX = max(map(lambda x: x[0], points))
            maxY = max(map(lambda x: x[1], points))
            midX = (minX + maxX) / 2
            midY = (minY + maxY) / 2
            points = [(round((point[0]-minX)/scale), round((point[1]-minY)/scale)) for point in points]
            yield points, (round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale))

def getShapes(infile: pya.Layout, layer: int,
              maxnum: int = None, verbose: bool = True):
    """
    Collects all shapes from layout into lists

    How it works:
        - uses yieldshape to iterate over shapes
        - collects shapes and their coordinates into lists
        - adjusts coordinates to ensure they are non-negative

    Args:
        infile: the input layout object
        layer: the layer to extract shapes from
        maxnum: maximum number of shapes to collect
        verbose: whether to print debug information

    Returns:
        two lists: polygons(list of shapes) and coords(list of bounding boxes)
    """
    iterator = yieldShape(infile, layer, verbose)
    polygons = []
    coords = []
    for datum, coord in iterator: 
        polygons.append(datum)
        coords.append(coord)
        if not maxnum is None and len(polygons) >= maxnum: 
            break
    minx = min(map(lambda x: x[0], coords))
    miny = min(map(lambda x: x[1], coords))
    coords = list(map(lambda x: (x[0]-minx, x[1]-miny, x[2]-minx, x[3]-miny, x[4]-minx, x[5]-miny), coords))
    for points in coords: 
        for elem in points: 
            assert elem >= 0, f"WRONG: {points}"
    
    return polygons, coords


def yieldShapes(infile, layer, fromX, fromY, toX, toY, anchor="min", verbose=True): # unit: nm
    ly = infile
    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell()
    topcell.flatten(-1)
    layer = ly.layer(layer, 0)
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu
    fromX = round(fromX * scale)
    fromY = round(fromY * scale)
    toX = round(toX * scale)
    toY = round(toY * scale)

    shapes = topcell.shapes(layer)
    for shape in shapes.each(): 
        polygons = []
        coords = []

        points = shape2points(shape, verbose)
        if len(points) > 0: 
            minX = min(map(lambda x: x[0], points))
            minY = min(map(lambda x: x[1], points))
            maxX = max(map(lambda x: x[0], points))
            maxY = max(map(lambda x: x[1], points))
            midX = (minX + maxX) / 2
            midY = (minY + maxY) / 2
            if anchor == "mid": 
                anchorX = midX
                anchorY = midY
            else: 
                anchorX = minX
                anchorY = minY
            reference = points
            countSkip = 0
            points = [(round((point[0]-anchorX)/scale), round((point[1]-anchorY)/scale)) for point in points]
            polygons.append(points)
            coords.append((round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale)))

            cbox = pya.Box.new(midX+fromX, midY+fromY, midX+toX, midY+toY)
            for neighbor in shapes.each_overlapping(cbox): 
                points = shape2points(neighbor, verbose)
                if len(points) > 0: 
                    if countSkip < 1 and len(points) == len(reference) and all(map(lambda x: points[x][0] == reference[x][0] and points[x][1] == reference[x][1], range(len(points)))): 
                        countSkip += 1
                        continue
                    minX = min(map(lambda x: x[0], points))
                    minY = min(map(lambda x: x[1], points))
                    maxX = max(map(lambda x: x[0], points))
                    maxY = max(map(lambda x: x[1], points))
                    midX = (minX + maxX) / 2
                    midY = (minY + maxY) / 2
                    points = [(round((point[0]-anchorX)/scale), round((point[1]-anchorY)/scale)) for point in points]
                    polygons.append(points)
                    coords.append((round(minX/scale), round(minY/scale), round(maxX/scale), round(maxY/scale), round(midX/scale), round(midY/scale)))
            assert countSkip == 1, f"ERROR: countSkip == {countSkip}"

        if len(polygons) > 0: 
            yield polygons, coords

def yieldCrops(infile, layer,
               sizeX, sizeY,
               strideX, strideY,
               offsetX=0, offsetY=0,
               fromzero=True, verbose=True): # unit: nm
    """
    Iterates over a layout, cropping it into smaller tiles

    How it works:
        1. Scaling coordiantes
            - convert input dimensions and strides into DBU (database units)
        2. Tiling
            - iterating over layout in steps strideX, strideY
            - crops the layout into tiles of (sizeX, sizeY)
        3. Shape extraction
            - extracts shapes from each tile and converts thenm to points
        4. Yielding
            - yields the polygons and coordinates of each tile

    Args:
        - infile: the input layout object
        - layer: the layer to extract
        - sizeX, sizeY: dimensions of each tile
        - strideX, strideY: step size for tiling
        - offsetX, offsetY: offset for the tiling grid
        - fromzero: whether to start tiling from (0,0)
        - verbose: whether to print debug information

    Yields:
        (polygons, (x,y)) for each tile
    """
    ly = infile
    print(f'Getting cells from a layout...')
    num_cells = ly.cells()

    # Iterate over all cells and print their names
    for cell_index in range(num_cells):
        cell = ly.cell(cell_index)
        cell_name = cell.name
        print(f"Cell {cell_index}: {cell_name}")

    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell().cell_index()
    layer = ly.layer(layer, 0)
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    if fromzero: 
        left = max(0, left)
        bottom = max(0, bottom)
    if verbose: 
        print(f"Bounding box: {bbox}, selected layer: {layer}")
        
    scale = 0.001 / ly.dbu
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    strideX = round(strideX * scale)
    strideY = round(strideY * scale)
    offsetX = round(offsetX * scale)
    offsetY = round(offsetY * scale)
    left += offsetX
    bottom += offsetY

    countTotal = 0
    # for idx in range(left - (strideX-1), right + (strideX - 1), strideX): 
    for idx in range(left, right + (strideX - 1), strideX): 
        if verbose: 
            print(f"X position: {round(idx/scale)} -> {round((idx+sizeX)/scale)} / {round(right/scale)}, Y range: {round(bottom/scale)} - {round(top/scale)}, count={countTotal}")
        # for jdx in range(bottom - (strideY-1), top + (strideY-1), strideY): 
        for jdx in range(bottom, top + (strideY-1), strideY): 
            countTotal += 1
            cbox = pya.Box.new(idx, jdx, idx + sizeX, jdx + sizeY)
            cropped = ly.clip(topcell, cbox)
            cell = ly.cell(cropped)
            cell.flatten(-1)
            cell.name = f"Cropped_{idx}-{jdx}_{idx+sizeX}-{jdx+sizeY}"
            print(cell.name)

            shapes = cell.shapes(layer)
            polygons = []
            for shape in shapes.each(): 
                points = shape2points(shape, verbose=False)

                points = [(round((point[0]-idx)/scale), round((point[1]-jdx)/scale)) for point in points]
                if len(points) > 0:
                    polygons.append(points)
            
            if len(polygons) > 0: 
                yield polygons, (round(idx/scale), round(jdx/scale))

def getCrops(infile, layer,
             sizeX, sizeY,
             strideX, strideY,
             offsetX=0, offsetY=0,
             maxnum=None, fromzero=True,
             verbose=True): # unit: nm
    """
    Based on yieldCrops function returns polygon coordinates inside every crop in a layout
    and (x,y) positions of the crops

    How it works:
        - gets an iterator with the help of yieldCrops
        - adds elements to empty lists

    Args:
        infile: the input layout object
        layer: the layer to extract
        sizeX, sizeY: dimensions of each tile
        strideX, strideY: step size for tiling
        offsetX, offsetY: offset for the tiling grid
        maxnum: maximum number of crops to process
        fromzero: whether to start tiling from (0,0)
        verbose: whether to print debug information

    Returns:
        list of polygon coordianates from layout crops and list of (x,y) positions of these crops
    """
    iterator = yieldCrops(infile, layer, sizeX, sizeY, strideX, strideY, offsetX=offsetX, offsetY=offsetY, fromzero=fromzero, verbose=verbose)
    images = []
    coords = []
    for datum, (x, y) in iterator: 
        images.append(datum)
        coords.append((x, y))
        if not maxnum is None and len(images) >= maxnum: 
            break
    return images, coords


if __name__ == "__main__": 
    import polygon as poly
    import matplotlib.pyplot as plt

    infile = pya.Layout()
    infile.read("benchmark/gcd_45nm.gds")
    print(f'Layout database unit: {infile.dbu}')
    cropped = cropLayout(infile, 11, 0, 31000, 10700, 23000)
    cropped.write('tmp/cropped_layout.gds')
    # yieldShape function, demonstrating on 5 shapes
    new_shape = yieldShape(infile=cropped, layer = 11, verbose=True)
    for i,shape in enumerate(new_shape):
        print(f'{i}-th shape of the polygon: {shape}')
        if i > 5:
            break
    shapes, coords = getShapes(cropped, layer = 11, maxnum=None, verbose=True)

    print(f'Shapes: {shapes}\n')
    print(f'Coords: {coords}\n')
    polygons = []
    for datum, coord in zip(shapes, coords):
        # adding minX and minY to normalized coordiantes to obtain the original values
        # want to obtain the original polygons to write a new layout file with
        # createLayout function
        polygon = list(map(lambda x: (x[0]+coord[0], x[1]+coord[1]), datum))
        dissected = poly.dissect(polygon, lenCorner=35, lenUniform=70)
        reconstr = poly.segs2poly(dissected)
        polygons.append(reconstr)
    plt.figure(figsize=(6,6))
    plt.title(f'Displaying reconstructed tile variant')
    plt.imshow(poly2imgShifted(polygons, sizeX=10000, sizeY=10000, scale=1))
    plt.show()
    print(f"In total {len(polygons)} shapes")
    layout = createLayout(polygons, layer=0, dbu=1e-3)
    cropped.write("tmp/test0.gds")
    layout.write("tmp/test1.gds")

    shapes, coords = getCrops(layout, layer=0, sizeX=2000, sizeY=2000, strideX=2000, strideY=2000, maxnum=None, verbose=True)
    count = 0
    maxnum = 0
    for datum, coord in zip(shapes, coords):
        count += 1
        if len(datum) > maxnum:
            maxnum = len(datum)
        print(f"Shapes: {coord}, {datum}")
        cv2.imwrite(f'tmp/example_{count}.jpg',poly2imgShifted(datum, sizeX=2000, sizeY=2000))
    print(f"Count = {count}")