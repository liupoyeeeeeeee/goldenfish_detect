import tkinter as tk
from tkinter import messagebox
import torch
import cv2
from tkinter import filedialog

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/berto/Desktop/三下/數值方法/期末project/yolov5/best.pt') 

def select_mode(mode):
    if mode == 'v':
        messagebox.showinfo("選擇", "你選擇了影片")
        video_window = tk.Toplevel(root)
        button = tk.Button(video_window, text="Select Video", command=select_video)
        button.pack()
    elif mode == 'p':
        messagebox.showinfo("選擇", "你選擇了照片")
        photo_window = tk.Toplevel(root)
        button = tk.Button(photo_window, text="Select Image", command=select_image)
        button.pack()

root = tk.Tk()
root.title("金魚辨識")

label = tk.Label(root, text="請選擇你要上傳的檔案")
label.pack()

video_button = tk.Button(root, text="影片", command=lambda: select_mode('v'))
video_button.pack()

photo_button = tk.Button(root, text="照片", command=lambda: select_mode('p'))
photo_button.pack()

# The rest of your code...


def select_image():
    # Open a file dialog and get the selected image file path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    
    # Inference
    results = model(img_path)

    # Results
    img = cv2.imread(img_path)
    for x1, y1, x2, y2, score, label in results.xyxy[0]:
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{label} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Resize the image to fit the screen
    screen_res = 1280, 720  # Change this to your screen resolution
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', window_width, window_height)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_video():
    # Open a file dialog and get the selected video file path
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while(cap.isOpened()):
        # Read the video frame by frame
        ret, frame = cap.read()
        if ret == True:
            # Inference
            results = model(frame)

            # Results
            for x1, y1, x2, y2, score, label in results.xyxy[0]:
                x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Resize the frame to fit the screen
            screen_res = 1280, 720  # Change this to your screen resolution
            scale_width = screen_res[0] / frame.shape[1]
            scale_height = screen_res[1] / frame.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(frame.shape[1] * scale)
            window_height = int(frame.shape[0] * scale)

            frame = cv2.resize(frame, (window_width, window_height))

            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

root.mainloop()
