# Program To Read video
# and Extract Frames
import cv2
import glob
import os


# Function to extract frames
def add_frame_numbers(path, file_name):
    # Path to video file
    video_in = cv2.VideoCapture(os.path.join(path, file_name + '.avi'))
    image_array = []

    # frame_number
    frame_number = 0

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # location
    location = (400, 100)

    # font_scale
    font_scale = 4

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2

    # size
    size = ()

    # checks whether frames were extracted
    success = True

    while success:
        frame_number += 1
        # video_in object calls read
        # function extract frames
        success, image = video_in.read()
        image_array.append(cv2.putText(image, str(frame_number), location, font,
                                       font_scale, color, thickness, cv2.LINE_AA))
        if frame_number == 1:
            height, width, layers = image.shape
            size = (width, height)
        # Saves the frames with frame-count
        # cv2.imwrite("frame%d.jpg" % count, image)

    out = cv2.VideoWriter(os.path.join(path, 'with_frame_nums/' + file_name + '_frames.avi'),
                                       cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()


# Driver Code
if __name__ == '__main__':
    # Calling the function
    video_path = 'data/GroundTruth/'
    file_names = glob.glob(os.path.join(video_path, '*.avi'))
    for full_file_name in file_names:
        file_name = '.'.join(full_file_name.split('/')[-1].split('.')[:-1])
        add_frame_numbers(video_path, file_name)
