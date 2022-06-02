from math import e
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cv2
from court_context import CourtContext, SIDE_UP, SIDE_BOTTOM

class PlayerData:
  def __init__(self, video_path, name, side, fps, bcc):
    self.video_path = video_path
    self.side = side
    self.name = name
    self.rawPositions = []
    self.rawPositions_img = []
    self.positions = []
    self.bcc = bcc

  def AddPosition(self, xy):
    if(xy != None):
      xy_court = self.bcc.getCourtPointFromImagePoint(xy)
      self.rawPositions.append(xy_court)
      self.rawPositions_img.append(xy)
    else:
      self.rawPositions.append(None)
      self.rawPositions_img.append(None)
