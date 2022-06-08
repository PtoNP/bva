import numpy as np
import pandas as pd
import os
import cv2
from players_positions.person_detection import DetectorAPI
from players_positions.court_context import CourtContext, SIDE_UP, SIDE_BOTTOM
from players_positions.player_data import PlayerData

class ExtractPlayersPositions:
  def __init__(self, video_path, match_name):
    self.match_name = match_name
    self.video_path = video_path
    self.fps = 0
    self.PlayerData_A = None
    self.PlayerData_B = None

  # define court middle lines
  def SetCourtLines(self, topMiddle, rightMiddle, bottomMiddle, leftMiddle):
    self.bcc = CourtContext()
    self.middles = np.array( [topMiddle,rightMiddle, bottomMiddle, leftMiddle], np.float32)
    self.bcc.setHomographyFromMiddles(self.middles)

  # define court corners
  def SetCourtCorners(self, upperLeftCorner, upperRightCorner,
                          bottomRightCorner, bottomLeftCorner):
    self.bcc = CourtContext()
    self.corners = np.array( [upperLeftCorner,upperRightCorner,
                         bottomRightCorner, bottomLeftCorner], np.float32)
    self.bcc.setHomographyFromCorners(self.corners)


  # main
  def Run(self, every_n_frames = 3, startTime = 0, endTime = None, noVideoOut=False):

    original = cv2.VideoCapture(self.video_path)
    self.fps = original.get(cv2.CAP_PROP_FPS)

    self.PlayerData_A = PlayerData(self.video_path,
                            "A", SIDE_UP, self.fps, self.bcc)
    self.PlayerData_B = PlayerData(self.video_path,
                            "B", SIDE_BOTTOM, self.fps, self.bcc)

    width = original.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = original.get(cv2.CAP_PROP_FRAME_HEIGHT )

    if noVideoOut == False:
      out = cv2.VideoWriter(f"{self.video_path[:-4]}_positions.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (int(width),int(height)))
      out_court = cv2.VideoWriter(f"{self.video_path[:-4]}_tracker.mp4", cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (int(610),int(1340)))
      #out_court = []

    odapi = DetectorAPI()
    threshold = 0.6

    has_next, i_frame = original.read()

    frame_counter = 0
    sequence_frame_counter = 0

    while has_next == True and (endTime == None or int(frame_counter/self.fps) < endTime):
      if int(frame_counter/self.fps) >= startTime:
        img_court = self.bcc.drawCourt()

        img_court = self.bcc.drawImagePositionOnCourt(img_court, self.corners[0])
        img_court = self.bcc.drawImagePositionOnCourt(img_court, self.corners[1])
        img_court = self.bcc.drawImagePositionOnCourt(img_court, self.corners[2])
        img_court = self.bcc.drawImagePositionOnCourt(img_court, self.corners[3])


        img = i_frame

        if frame_counter % every_n_frames == 0:
            boxes, scores, classes, num = odapi.processFrame(img)
            centers_img = []
            centers_court = []

            for i in range(len(boxes)):
                #if classes[i] == 1 and scores[i] > threshold:
                if classes[i] == 1:
                    box = boxes[i]
                    img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)  # cv2.FILLED

                    img = cv2.putText(img, str(i), (box[1], box[0]),
                     cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)

                    if box[3]-box[1] < 150:
                        player_center_img = [box[2], int(box[1] + (box[3]-box[1]) / 2)]
                        player_center_court = self.bcc.getCourtPointFromImagePoint(player_center_img)

                        if self.bcc.positionInCourt(player_center_court):
                            img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                            centers_img.append(player_center_img)
                            centers_court.append(player_center_court)

            if(len(centers_court) >= 2):
                add_A = False
                add_B = False
                idx_box_A = self.bcc.closestPointToTopT(centers_court)
                center_A = centers_img[idx_box_A]
                add_A = centers_court[idx_box_A][0] >= self.bcc.up_left_corner[0] \
                    and centers_court[idx_box_A][0] <= self.bcc.middle_left[0]
                del centers_court[idx_box_A]
                del centers_img[idx_box_A]

                idx_box_B = self.bcc.closestPointToBottomT(centers_court)
                center_B = centers_img[idx_box_B]

                add_B = centers_court[idx_box_B][0] >= self.bcc.middle_left[0] \
                    and centers_court[idx_box_B][0] <= self.bcc.bottom_left_corner[0]
                del centers_court[idx_box_B]
                del centers_img[idx_box_B]

                if add_A and add_B:
                    self.PlayerData_A.AddPosition(center_A)
                    self.PlayerData_B.AddPosition(center_B)
                else:
                    self.PlayerData_A.AddPosition(None)
                    self.PlayerData_B.AddPosition(None)


                img_court = self.bcc.drawImagePositionOnCourt(img_court, center_A)
                img_court = self.bcc.drawImagePositionOnCourt(img_court, center_B)

            else:
                self.PlayerData_A.AddPosition(None)
                self.PlayerData_B.AddPosition(None)
        else:
            self.PlayerData_A.AddPosition(None)
            self.PlayerData_B.AddPosition(None)

        if noVideoOut == False:
            out.write(img.astype('uint8'))
            #out_court.append(img_court.astype('uint8'))
            out_court.write(img_court.astype('uint8'))

        sequence_frame_counter += 1

      frame_counter += 1
      has_next, i_frame = original.read()

    if noVideoOut == False:
        out.release()
        out_court.release()

    self.ExportPlayersPositionsToCSV(f"{self.video_path[:-4]}_players.csv")

    print(f' Positions of {self.video_path} exported')

  def ExportPlayersPositionsToCSV(self, filename):
    textfile = open(filename, "w")
    counter_frame = 0
    textfile.write(f'video_path,frame,player_A_visible,player_B_visible,player_A_court_x,player_A_court_y,player_A_img_x,player_A_img_y,player_B_court_x,player_B_court_y,player_B_img_x,player_B_img_y\n')

    for i in range(len(self.PlayerData_A.rawPositions)):
        if self.PlayerData_A.rawPositions[i]:
            player_A_visible = 1
            player_B_visible = 1
            player_A_court_x = self.PlayerData_A.rawPositions[i][0]
            player_A_court_y = self.PlayerData_A.rawPositions[i][1]
            player_A_img_x = self.PlayerData_A.rawPositions_img[i][0]
            player_A_img_y = self.PlayerData_A.rawPositions_img[i][1]
            player_B_court_x = self.PlayerData_B.rawPositions[i][0]
            player_B_court_y = self.PlayerData_B.rawPositions[i][1]
            player_B_img_x = self.PlayerData_B.rawPositions_img[i][0]
            player_B_img_y = self.PlayerData_B.rawPositions_img[i][1]
        else:
            player_A_visible = 0
            player_B_visible = 0
            player_A_court_x = -1
            player_A_court_y = -1
            player_A_img_x = -1
            player_A_img_y = -1
            player_B_court_x = -1
            player_B_court_y = -1
            player_B_img_x = -1
            player_B_img_y = -1

        textfile.write(f'{self.match_name},{counter_frame}, \
                         {player_A_visible},{player_B_visible}, \
                         {player_A_court_x},{player_A_court_y}, \
                         {player_A_img_x},{player_A_img_y}, \
                         {player_B_court_x},{player_B_court_y}, \
                         {player_B_img_x},{player_B_img_y}\n')

        counter_frame += 1

    textfile.close()

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    match_path = 'match2/rally_video/1_00_02.mp4'
    #match_path = 'match9/rally_video/1_07_11.mp4'
    video_path = f'{cur_dir}/../../raw_data/01_TRAIN/{match_path}'

    video_details_path = f'{cur_dir}/../../raw_data/video_details.csv'
    video_details = pd.read_csv(video_details_path)

    details = video_details[video_details['video_path'] == match_path]

    epp = ExtractPlayersPositions(video_path, match_path)
    epp.SetCourtCorners(
        [details.iloc[0]['ul_corner_y'], details.iloc[0]['ul_corner_x']],
        [details.iloc[0]['ur_corner_y'], details.iloc[0]['ur_corner_x']],
        [details.iloc[0]['br_corner_y'], details.iloc[0]['br_corner_x']],
        [details.iloc[0]['bl_corner_y'], details.iloc[0]['bl_corner_x']]
    )

    epp.Run(every_n_frames=1, endTime=1)
