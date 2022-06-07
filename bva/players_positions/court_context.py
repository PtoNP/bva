import numpy as np
import cv2

SIDE_UP = 1
SIDE_BOTTOM = 2

class CourtContext:
  def __init__(self):
    self.up_left_corner = [0,0]
    self.up_right_corner = [0,610]
    self.up_middle = [0,305]
    self.up_service_left = [470,0]
    self.up_service_right = [470,610]
    self.up_service_middle = [470,305]
    self.middle_left = [670,0]
    self.middle_right = [670,610]
    self.middle_middle = [670,305]
    self.bottom_service_left = [870,0]
    self.bottom_service_right = [870,610]
    self.bottom_service_middle = [870,305]
    self.bottom_right_corner = [1340,610]
    self.bottom_left_corner = [1340,0]
    self.bottom_middle = [1340,305]

    self.up_serve_even = [250, 270]
    self.up_serve_odd  = [250, 340]
    self.bottom_receive_even = [1070,405]
    self.bottom_receive_odd  = [1070,205]

    self.up_receive_even = [250, 205]
    self.up_receive_odd  = [250, 405]
    self.bottom_serve_even = [1070,340]
    self.bottom_serve_odd  = [1070,270]

  def setHomographyFromCorners(self, img_points):
    # img_points = [top_left_corner[y,x], top_right_corner[y,x], bottom_right_corner[y,x], bottom_left_corner[y,x]]
    pts_src = np.array(
            [self.up_left_corner,
            self.up_right_corner,
            self.bottom_right_corner,
            self.bottom_left_corner],
            np.float32)

    h, status = cv2.findHomography(pts_src, img_points, cv2.RANSAC)
    self.courtToImageTransform = h

    h, status = cv2.findHomography(img_points, pts_src, cv2.RANSAC)
    self.imageToCourtTransform = h


  def setHomographyFromMiddles(self, img_points):
    # img_points = [middle_top[y,x], middle_right[y,x], middle_bottom[y,x], middle_left[y,x]]
    pts_src = np.array(
            [self.up_middle,
            self.middle_right,
            self.bottom_middle,
            self.middle_left],
            np.float32)

    h, status = cv2.findHomography(pts_src, img_points, cv2.RANSAC)
    self.courtToImageTransform = h

    h, status = cv2.findHomography(img_points, pts_src, cv2.RANSAC)
    self.imageToCourtTransform = h

  def getImagePointFromCourtPoint(self, p):
    point = np.array([p], np.float32)
    result = cv2.perspectiveTransform(point[None, :, :], self.courtToImageTransform)
    return [int(result[0][0][0]),int(result[0][0][1])]

  def getCourtPointFromImagePoint(self, p):
    point = np.array([p], np.float32)
    result = cv2.perspectiveTransform(point[None, :, :], self.imageToCourtTransform)
    return [int(result[0][0][0]),int(result[0][0][1])]

  def closestNode(self, node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

  def positionInCourt(self, p):
    return p[1] >= self.up_left_corner[1] \
      and  p[1] <= self.up_right_corner[1] \
      and  p[0] >= self.up_left_corner[0] \
      and  p[0] <= self.bottom_left_corner[0]

  def closestPointToTopT(self, pts):
    return self.closestNode(self.up_service_middle, pts)

  def closestPointToBottomT(self, pts):
    return self.closestNode(self.bottom_service_middle, pts)

  def drawCourt(self):
    width = int(self.up_right_corner[1] - self.up_left_corner[1])
    height = int(self.bottom_left_corner[0] - self.up_left_corner[0])
    img = np.zeros((height,width,3), np.uint8)
    img[:,:,:] = (39,137,80)

    cv2.line(img, (self.up_left_corner[1], self.up_left_corner[0]), (self.up_right_corner[1], self.up_right_corner[0]), (255, 255, 255), 2)
    cv2.line(img, (self.up_service_left[1], self.up_service_left[0]), (self.up_service_right[1], self.up_service_right[0]), (255, 255, 255), 2)
    cv2.line(img, (self.bottom_service_left[1], self.bottom_service_left[0]), (self.bottom_service_right[1], self.bottom_service_right[0]), (255, 255, 255), 2)
    cv2.line(img, (self.bottom_left_corner[1], self.bottom_left_corner[0]), (self.bottom_right_corner[1], self.bottom_right_corner[0]), (255, 255, 255), 2)

    cv2.line(img, (self.up_left_corner[1], self.up_left_corner[0]), (self.bottom_left_corner[1], self.bottom_left_corner[0]), (255, 255, 255), 2)
    cv2.line(img, (self.up_middle[1], self.up_middle[0]), (self.up_service_middle[1], self.up_service_middle[0]), (255, 255, 255), 2)
    cv2.line(img, (self.bottom_middle[1], self.bottom_middle[0]), (self.bottom_service_middle[1], self.bottom_service_middle[0]), (255, 255, 255), 2)
    cv2.line(img, (self.up_right_corner[1], self.up_right_corner[0]), (self.bottom_right_corner[1], self.bottom_right_corner[0]), (255, 255, 255), 2)

    cv2.line(img, (self.middle_left[1], self.middle_left[0]), (self.middle_right[1], self.middle_right[0]), (0, 0, 0), 3)

    cv2.circle(img, (self.up_service_middle[1],self.up_service_middle[0]), 5, (255, 0, 0), -1)
    cv2.circle(img, (self.bottom_service_middle[1],self.bottom_service_middle[0]), 5, (255, 0, 0), -1)

    return img

  def drawImagePositionOnCourt(self, img, imagePosition, radius = 10, color = (0, 0, 255), filled=False):
    courtPosition = self.getCourtPointFromImagePoint(imagePosition)
    if filled:
        thickness = -1
    else:
        thickness = 2
    cv2.circle(img, (courtPosition[1],courtPosition[0]), radius, color, thickness)
    return img

  def drawCourtPosition(self, img, courtPosition, radius = 10, color = (0, 0, 255), filled=False):
    if filled:
        thickness = -1
    else:
        thickness = 2
    cv2.circle(img, (courtPosition[1],courtPosition[0]), radius, color, thickness)
    return img
