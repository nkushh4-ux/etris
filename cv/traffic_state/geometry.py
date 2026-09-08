import cv2
import numpy as np

from cv.detection.models import BoundingBox
from cv.traffic_state.models import Point


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    return cv2.pointPolygonTest(np.asarray(polygon,dtype=np.float32), point, False) >= 0


def line_side(point: Point, line: tuple[Point, Point]) -> float:
    (ax,ay),(bx,by)=line
    return (bx-ax)*(point[1]-ay)-(by-ay)*(point[0]-ax)


def segments_intersect(first: tuple[Point,Point], second: tuple[Point,Point]) -> bool:
    a,b=first; c,d=second
    values=(line_side(c,(a,b)),line_side(d,(a,b)),line_side(a,(c,d)),line_side(b,(c,d)))
    return values[0]*values[1] <= 0 and values[2]*values[3] <= 0


def moves_in_direction(previous:Point,current:Point,direction:Point) -> bool:
    return (current[0]-previous[0])*direction[0]+(current[1]-previous[1])*direction[1] > 0


def polygon_mask(polygon: tuple[Point,...], width: int, height: int) -> np.ndarray:
    points=np.asarray([((min(1,max(0,x))*(width-1)),(min(1,max(0,y))*(height-1))) for x,y in polygon],dtype=np.int32)
    mask=np.zeros((height,width),dtype=np.uint8); cv2.fillPoly(mask,[points],1); return mask


def bbox_coverage_area(bbox: BoundingBox, mask: np.ndarray) -> int:
    height,width=mask.shape; x1=max(0,min(width,int(bbox.x1))); y1=max(0,min(height,int(bbox.y1)))
    x2=max(0,min(width,int(np.ceil(bbox.x2)))); y2=max(0,min(height,int(np.ceil(bbox.y2))))
    return int(mask[y1:y2,x1:x2].sum()) if x2>x1 and y2>y1 else 0


def union_bbox_occupancy(bboxes: tuple[BoundingBox,...], polygon: tuple[Point,...], width:int, height:int) -> float:
    region=polygon_mask(polygon,width,height); area=int(region.sum())
    if not area: return 0.0
    covered=np.zeros_like(region)
    for bbox in bboxes:
        x1=max(0,min(width,int(bbox.x1))); y1=max(0,min(height,int(bbox.y1)))
        x2=max(0,min(width,int(np.ceil(bbox.x2)))); y2=max(0,min(height,int(np.ceil(bbox.y2))))
        if x2>x1 and y2>y1: covered[y1:y2,x1:x2]=1
    return max(0.0,min(1.0,float(np.logical_and(covered,region).sum()/area)))
