import os
import sys
import math
import copy
import time
import pickle
import random
import argparse

import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

def poly2img(polygons: List[List[Tuple[int, int]]],
             sizeX: int,
             sizeY: int,
             scale: float = 1.0) -> np.ndarray:
    """
    Converts a list of polygons into a 2D image representation.

    How it works:
        - creates a blank image
        - fills the defined polygons using OpenCV's cv2.fillPolly().

    Args:
        polygons: a list of polygons where each polygon is a list of (x,y) coordinates
        sizeX, sizeY: Image dimensions
        scale: A scaling factor for resizing polygons

    Returns:
        np.ndarray: an image with drawn polygons

    Note:
        if we want to test on a single polygons, we need to wrap the passed variable in a list, f.e.

        .. code-block::

            poly2img(polygons=[polygon], sizeX=, sizeY=, scale=),

        where polygon=[(), (), ()]
    """
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    img = np.zeros([sizeY, sizeX], dtype=np.float32)
    for idx in range(len(polygons)): 
        polygon = np.array(polygons[idx])
        polygon = np.round(polygon * scale).astype(np.int64)
        img = cv2.fillPoly(img, [polygon], color=255)
    return img

def polysMin(polygons: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Finds the minimum x and y coordinates among all polygons.

    How it works:
        Iterates over all polygon points to compute the smallest x and y values

    Args:
        polygons: a list of polygons.

    Returns:
        a tuple (minX, minY) with smallest coordinates among all polygons
    """
    minX = None
    minY = None
    for idx in range(len(polygons)): 
        min1 = min(map(lambda x: x[0], polygons[idx]))
        min2 = min(map(lambda x: x[1], polygons[idx]))
        minX = min1 if minX is None else min(minX, min1)
        minY = min2 if minY is None else min(minY, min2)
    return (minX, minY)
def poly2imgShifted(polygons:List[List[Tuple[int, int]]],
                    sizeX:int,
                    sizeY:int,
                    scale:int = 1.0,
                    shifted: Tuple[int, int] = None) -> np.ndarray:
    """
    Draws polygons on image after shifting them by their minimum coordinates

    How it works:
    - Translates polygons by subracting (minX, minY) from each point in every polygon
    - Fills the translated polygons on the image

    Args:
        polygons: a list of polygons where each polygon is a list of (x,y) coordinates
        sizeX, sizeY: image dimensions
        scale: a scaling factor for resizing polygons
        shifted: an optional parameter to define a custom shift instead on shifting
        coordinates by (minX, minY)

    Returns:
        a shifted and reversed image

    Note:
        img[::-1, :] - flips the image vertically (along the y-axis),
        reverses the row order but keeps the columns intact
    """
    sizeX = round(sizeX * scale)
    sizeY = round(sizeY * scale)
    img = np.zeros([sizeY, sizeX], dtype=np.float32)
    minX, minY = polysMin(polygons) if shifted is None else shifted
    minval = np.array([[minX, minY, ], ])
    for idx in range(len(polygons)): 
        polygon = np.array(polygons[idx]) - minval
        polygon = np.round(polygon * scale).astype(np.int64)
        img = cv2.fillPoly(img, [polygon], color=255)
    return img[::-1, :]

def lines(polygon: List[Tuple[int, int]]
          ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    '''
    Breaks a polygon into its line segments. Connects each vertex to the next,
    closing a polygon

    Args:
        polygon: a list of points representing the polygon vertices

    Note:
        Function works ONLY with a single polygon, f.e. [(x1, y1), (x2,y2), (x3,y3)]

    Returns:
        a list of line segments as tuples ((x1,y1), (x2,y2))

    Example:
        .. parsed-literal::

            For polygon [(500, 500), (500, 1000), (700, 1000), (700, 500)]

            function returns:

            [((500, 500), (500, 1000)), ((500, 1000), (700, 1000)), ((700, 1000), (700, 500)), ((700, 500), (500, 500))]
    '''
    results = []
    for idx in range(len(polygon)): 
        results.append((polygon[idx], polygon[(idx+1)%len(polygon)]))
    return results
def dissect(polygon: List[Tuple[int, int]],
            lenCorner: int,
            lenUniform: int,
            verbose: bool = False
            ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    '''
    Subdivies the edges of a polygon into smaller line segments.

    How it works:
        - splits edges based on whether they are vertical or horizontal
        - ensures that large segments are divided into smaller, equal-length parts

    Args:
        polygon: a list of points forming a polygon
        lenCorner: corner segment length
        lenUniform: uniform segment length for dividing long edges

    Returns:
        a list of dissected segments

    Note:
        .. parsed-literal::
            The code explanation will be provided for the polygon defined as follows:

            [(500, 500), (500, 1000), (700, 1000), (700,500)]

            (500, 500)  <-  (700, 500)
                |               |
            (500, 1000) -> (700, 1000)

        We will observe the first line segment - ((500, 500), (500, 1000))
    '''
    # 1. Initial setup
    # Results will store the final list of dissected polygon segments.
    # The function loops through all lines (edges) of the polygon
    results = []
    for line in lines(polygon): 
        segments = []
        # 2. Axis Determination
        # the block asserts that the line is either vertical or horizontal, as required for
        # processing simple polygons.
        # For each edge:
        # (500, 500) -> (500, 1000) is vertical, so axis = 1
        # (500, 1000) -> (700, 1000) is horizontal, so axis = 0
        assert line[0][0] == line[1][0] or line[0][1] == line[1][1]
        if line[0][0] == line[1][0]: 
            axis = 1
        elif line[0][1] == line[1][1]: 
            axis = 0
        # 3. Identifying Start and End points  of the segment and assigns the correct order
        # to endpoints (smaller, bigger).
        # For (500, 500) -> (500, 1000):
        # smaller = 0, bigger = 1, and length = 500.
        if line[0][axis] > line[1][axis]: 
            smaller = 1
            bigger = 0
        else: 
            smaller = 0
            bigger = 1
        length = line[bigger][axis] - line[smaller][axis]
        assert length > 0

        if length < 2 * lenCorner:
            # 4. Handling Short Segments
            # If the line is too short (less than 2 * lenCorner),
            # it isn’t dissected and is directly added as a segment.
            segments.append(line)
        else:
            # 5. Defining Corner and Uniform Segments
            # This defiend two "inner" points by trimming lenCorner from each end
            # For segment (500, 500) -> (500, 1000):
            # start = 530, end = 970
            # point1 = (500, 530), point2  = (500, 970)
            start = line[smaller][axis] + lenCorner
            end = line[bigger][axis] - lenCorner
            point1 = list(line[smaller])
            point1[axis] = start
            point1 = tuple(point1)
            point2 = list(line[bigger])
            point2[axis] = end
            point2 = tuple(point2)

            # 6. Creating Uniform Segments
            # This loop creates dissected segments between the start and midpoint of the edge
            # For vertical line (500, 530) -> (500, 970):
            # uniform segment length is 50, so points are created at positions 580, 630, etc

            last1 = point1
            middle = start + round((end - start) / 2)
            segments1 = []
            segments1.append((line[smaller], point1))
            for pos1 in range(start+lenUniform, middle, lenUniform): 
                point = list(point1)
                point[axis] = pos1
                point = tuple(point)
                segments1.append((last1, point))
                last1 = point

            # Similarly, segments are created on the other half of the line (segments2)
            last2 = point2
            segments2 = []
            segments2.append((point2, line[bigger]))
            for pos2 in range(end-lenUniform, middle, -lenUniform): 
                point = list(point2)
                point[axis] = pos2
                point = tuple(point)
                segments2.append((point, last2))
                last2 = point
            # 7. Handling Short Middle Segments
            # If gap between last1 (last point from segments1) and last2 (last point from segments2)
            # is too small to fit lenUniform, then two closest segments are merged.
            # F.e., last1 = (500, 730), last2 = (500, 770), axis = 1
            # The first condition is True ->
            # segment1 = ((500, 680), (500, 730))
            # segment2 = ((500, 770), (500, 820))
            # new segment ((500, 680), (500, 820)) is added to segments1
            # In case first condition was False,
            # we would add ((500, 730), (500, 770)) to segments1
            if last2[axis] - last1[axis] < lenUniform: 
                segment1 = segments1.pop()
                segment2 = segments2.pop()
                segments1.append((segment1[0], segment2[1]))
            else: 
                segments1.append((last1, last2))
            # adding segments from 1-st [start, mid_point) and 2-nd (mid_point, end] groups
            # to the main list
            segments.extend(segments1)
            # segments from second group are converted to
            # the normal format, when end of i-th segment equals start of the i+1-th segment
            segments.extend(segments2[::-1])

            for segment in segments: 
                assert abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]) >= min(lenCorner, lenUniform)
        # 8.
        # 8.1 Condition: checks if the current line's first point (line[0]) is "greater" than its second point (line[1])
        # along the chosen axis (axis=0 for horizontal lines or axis = 1 for vertical lines)
        # This means the points are listed in reverse order from top to bottom or left to right.
        # 8.2 Reversing the order: segments[::-1] reverses the list of segments
        # 8.3 Swapping points: map(lambda x: (x[1], x[0]), segments) swaps the start and end point of each segment
        # Why This is Needed ?
        # The goal here is to ensure that all segments consistently run from the lower to higher coordinate values
        # (top to bottom for vertical lines, left to right for horizontal lines). Without this block segments would
        # be mixed, making further geometric processing ambiguos
        if line[0][axis] > line[1][axis]: 
            segments = list(map(lambda x: (x[1], x[0]), segments[::-1]))
        
        results.extend(segments)

    # 9. Visualize the segment points on the raw polygon
    if verbose:
        x_list = []
        y_list = []
        for segment in results:
            x_list.extend([segment[0][0],segment[1][0] ])
            y_list.extend([segment[0][1], segment[1][1] ])

        print(f'X-coordiantes:{x_list}')
        print(f'Y-coordinates:{y_list}')

        plt.figure(figsize=(6,6))
        plt.imshow(poly2img([polygon], sizeX=4500, sizeY=4500, scale=1))
        plt.title(f'Segmented polygon')
        plt.scatter(x = np.array(x_list),y = np.array(y_list), s = 2)
        plt.show()

    return results

def segs2poly(segments: List[Tuple[Tuple[int, int], Tuple[int, int]]]
              ) -> List[Tuple[int, int]]:
    '''
    Reconstructs a closed polygon from line segments

    How it works:
        - checks for continuity between segments (horizontal or vertical)
        - corrects point alignments to form a valid closed polygon

    Args:
        segments: a list of segments

    Returns:
        a list of polygon vertices in proper order
    '''
    # 1. Create a deep copy of segments to avoid modifying the original list
    # Deep copying ensures that changes to legalized do not affect segments
    legalized = copy.deepcopy(segments)

    for idx in range(len(legalized)):
        # 2. Checking if segment on current/previous iteration is
        # horizontal (lastH/thisH) or vertical (lastV/thisV)
        # got 2 possible scenarios:
        #   (a)lastH and thisV
        #   (b)lastV and thisH
        lastH = legalized[idx-1][0][0] == legalized[idx-1][1][0]
        lastV = legalized[idx-1][0][1] == legalized[idx-1][1][1]
        assert (lastH or lastV)
        thisH = legalized[idx][0][0] == legalized[idx][1][0]
        thisV = legalized[idx][0][1] == legalized[idx][1][1]
        assert (thisH or thisV)
        # 3. Point reallignment
        # Here we ensure the continuity between segments
        # If the previous segment is horizontal and the current segment is vertical (lastH and thisV),
        # the end point of the horizontal segment is adjusted to
        # align with the starting point of the vertical one.

        if lastH and thisV:
            legalized[idx - 1] = (legalized[idx - 1][0], (legalized[idx - 1][1][0], legalized[idx][0][1]))
            legalized[idx] = ((legalized[idx - 1][1][0], legalized[idx][0][1]), legalized[idx][1])
        elif lastV and thisH:
            legalized[idx - 1] = (legalized[idx - 1][0], (legalized[idx][0][0], legalized[idx - 1][1][1]))
            legalized[idx] = ((legalized[idx][0][0], legalized[idx - 1][1][1]), legalized[idx][1])

    # 4. Constructing the polygon
    # Purpose: Combines the adjusted segments into a list of unique polygon vertices
    # The loop ensures that no consecutive duplicate vertices are included
    # polygon[:-1] removes the duplicate of the first point to ensure closure without redundancy.
    polygon = []
    for elems in legalized:
        for elem in elems:
            if len(polygon) == 0:
                polygon.append(elem)
            elif elem[0] != polygon[-1][0] or elem[1] != polygon[-1][1]:
                polygon.append(elem)
    return polygon[:-1]

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # polygon = [(3090, 15550), (3090, 15755), (2910, 15755), (2910, 16650), (2980, 16650), (2980, 15825), (3160, 15825), (3160, 15550)]
    #
    # dissected = dissect(polygon, lenCorner=35, lenUniform=70)
    # reconstr = segs2poly(dissected)
    # print(dissected)
    # print(reconstr)
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(poly2img([polygon], 20480, 20480, 0.1))
    # plt.subplot(1, 2, 2)
    # plt.imshow(poly2img([reconstr], 20480, 20480, 0.1))
    # plt.show()

    polygon = [[
        (1000, 1000), (3000, 1000), (3000, 3000), (2000, 3000),
        (2000, 2000), (1000, 2000)
    ],
    [(500, 500), (500, 1000), (700, 1000), (700,500)]]
    print(f'Raw polygons:{polygon}')
    print(f'(x_min, y_min): {polysMin(polygon)}')
    print(f'Line segments:{lines(polygon=polygon[1])}')

    plt.figure(figsize=(6,6))
    plt.title(f'Original image after applying poly2img function')
    plt.imshow(poly2img(polygon, sizeX=5000, sizeY=5000, scale=1))
    plt.show()

    plt.figure(figsize=(6,6))
    plt.title('Shifted and flipped along the y-axis image after applying poly2imgShifted')
    plt.imshow(poly2imgShifted(polygons = polygon, sizeX = 5000, sizeY = 5000, shifted = (400, 400)))
    plt.show()

    dissected = dissect(polygon[0], lenCorner = 30, lenUniform = 50, verbose = True)
    print(f'Dissected polygon: {dissected}')
    reconstructed = segs2poly(dissected)
    print(f'Reconstructed polygon: {reconstructed}')