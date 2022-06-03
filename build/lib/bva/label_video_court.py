import cv2
import os
import csv

results = {}
current_file_path = ''


def label_all_mp4(src_folder):
    count = 0
    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in [f for f in filenames if f.endswith(".mp4")]:
            #if count > 0:
            #    break
            label_court(os.path.join(dirpath, filename))
            count = count + 1

    fields = ['video_path', 'fps', 'width', 'height', \
                'ul_corner_x', 'ul_corner_y',
                'ur_corner_x', 'ur_corner_y',
                'br_corner_x', 'br_corner_y',
                'bl_corner_x', 'bl_corner_y',
                'left_net_x', 'left_net_y',
                'right_net_x', 'right_net_y'
             ]
    with open('courts.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        for result in results:
            write.writerow(results[result])


def draw_points(event, x, y, flags, frame):
    global current_file_path
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = cv2.circle(img=frame, center = (x,y), radius =3, color =(255,0,0), thickness=3)
        results[current_file_path].append(x)
        results[current_file_path].append(y)
        cv2.imshow('win', frame)


def label_court(video_file_path):
    global current_file_path
    current_file_path = video_file_path
    base=os.path.basename(video_file_path)
    file = os.path.splitext(base)[0]
    dir = os.path.dirname(os.path.abspath(video_file_path))

    stream = cv2.VideoCapture(video_file_path)
    fps = stream.get(cv2.CAP_PROP_FPS)
    length = stream.get(cv2.CAP_PROP_FRAME_COUNT)

    has_next, frame = stream.read()
    i = 0
    while has_next == True:
        if i == int(fps):
            break
        i = i + 1
        has_next, frame = stream.read()

    results[video_file_path] = [video_file_path, fps, frame.shape[1], frame.shape[0]]

    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('win', draw_points, frame)
    cv2.imshow('win', frame)

    cv2.waitKey(0)
    print(results[video_file_path])

if __name__ == '__main__':
    label_all_mp4('.\\03_INPUT')
