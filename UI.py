import tkinter as tk
from tkinter import *
import tkinter.font

import cv2
from PIL import Image,ImageTk
import os
import time
import glob
import torch
import torchvision
#print('UI.py',torch.__version__)
#print('UI.py', torch.__path__)
#print('UI.py',torchvision.__version__)
#print('UI.py', torchvision.__path__)

from YOLOv5.detect import YOLOdetect


def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code"""
    return "#%02x%02x%02x" % rgb   

def Close_UI(root):
    print('-----------------------------------------')
    print('Close the UI')
    root.destroy()

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    cam_ids = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cam_ids.append(index)
            cap.release()
        index += 1
        i -= 1
    print('returnCameraIndexes()=', cam_ids)
    return cam_ids


def take_snapshot(save_path):
    print('-----------------------------------------')
    for ID in All_frames:
        frame =  All_frames[ID]
        #frame = np.asarray(All_frames[i]) # Image类型 to numpy.ndarray'
        if not os.path.exists(save_path):
            os.mkdir(save_path)        
        cv2.imwrite(save_path + (str(ID)+'.jpg'), frame)
    print("Take snapshot")


def Begin():
    global stop
    print('-----------------------------------------')
    print("Begin video loop")
    stop=False
        

def Stop():
    global video, stop
    if stop == False:
        print("Stop video loop")
        stop=True
    else:
        pass


def ShowPredict(path):
    images_path = glob.glob(path)
    print('Predict_images_path=', images_path)
    current_image1 = Image.open(images_path[0])
    current_image2 = Image.open(images_path[1])
    current_image1 = current_image1.resize((640, 360))
    current_image2 = current_image2.resize((640, 360))
    left_canvas.image = ImageTk.PhotoImage(image=current_image1)
    right_canvas.image = ImageTk.PhotoImage(image=current_image2)
    
    global new_left_canvas
    global new_right_canvas
    left_canvas.itemconfig(new_left_canvas, image = left_canvas.image)
    right_canvas.itemconfig(new_right_canvas, image = right_canvas.image)


# listbox
def Location(Pred_loaction):
    images_path = glob.glob('Detection results/*.jpg')
    L_image_name = os.path.split(images_path[0])[1]
    R_image_name = os.path.split(images_path[1])[1]
    L_locs = Pred_loaction[L_image_name].split('/')
    R_locs = Pred_loaction[R_image_name].split('/')
    
    Left_listbox.delete(0, tk.END)  # 清空列表
    Right_listbox.delete(0, tk.END)  # 清空列表
    Left_listbox.insert(0, L_image_name)
    Right_listbox.insert(0, R_image_name)
    line = 1
    for i in range(len(L_locs)):
        if L_locs[i] == "":
            pass
        else:
            Left_listbox.insert(line, L_locs[i])
            line += 1

    for i in range(len(R_locs)):
        if R_locs[i] == "":
            pass
        else:            
            Right_listbox.insert(line, R_locs[i])
            line += 1
     
   

def video_loop():
    global stop
    #print('stop=', stop)
    val_cam_id = []
    if (stop is False):
        cam_ids = [0, 1]
        cameras = [cv2.VideoCapture(i) for i in cam_ids]
        for i in range(2):
            val_cam_id.append(cam_ids[i])
            cameras[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cameras[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame  = cameras[i].read()
            if ret:
                All_frames[cam_ids[i]] = frame
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # 轉換顏色從BGR到RGBA
                current_image = Image.fromarray(frame) # 將影象轉換成Image物
                imgtk = ImageTk.PhotoImage(image=current_image)
                All_images[str(cam_ids[i])] = imgtk
                
                if i == 0:
                    global new_left_canvas
                    new_left_canvas = left_canvas.create_image(0, 0, anchor=tk.NW, image=All_images[str(val_cam_id[0])])
                elif i == 1:
                    global new_right_canvas
                    new_right_canvas = right_canvas.create_image(0, 0, anchor=tk.NW, image=All_images[str(val_cam_id[1])])
             #root.update()
             
        # 拍照並儲存照片
        save_path = 'Photographed images/'
        take_snapshot(save_path)
        
        # YOLOv5 與 計算座標
        Location(YOLOdetect(save_img=False, nosave=False, project='Detection results',
                            exist_ok=True, device='cpu', source='Photographed images', 
                            weights=['./YOLOv5/runs/train/7_496124/weights/best.pt'],
                            view_img=False, save_txt=False, imgsz=640, name='',
                            augment=False, conf_thres=0.25, iou_thres=0.3, 
                            classes=None, agnostic_nms=False, save_conf=False, 
                            update=False))
        # 顯示預測結果
        ShowPredict(path='Detection results/*.jpg')
        root.update()
        
        global video
        video = root.after(100 , video_loop()) # 110毫秒之後重跑此function #video_loop慢且有值#video_loop()快且無值
    else:
        print('-----------------------------------------')
        print('Release the camera')
        # 當一切都完成後，關閉攝像頭並釋放所佔資源
        cameras = [cv2.VideoCapture(i) for i in cam_ids]
        for camera in cameras:
            camera.release()
        pass


# 設定window的工作列圖標
import ctypes 
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string 
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


global All_frames, All_images

All_frames = {}
All_images = {}
video = None
stop = False    

cam_ids = returnCameraIndexes()

root = tk.Tk()
root.title("跌倒辨識使用者介面")
root.geometry('1400x750+0+0')
# 設定tkinter左上角的工作列圖標
root.iconbitmap('front.ico')

Word_type ='微軟正黑體'
boldFont = tkinter.font.Font(family=Word_type, size=23, weight="bold") # 達成粗體文字呈現
btn_boldFont = tkinter.font.Font(family=Word_type, size=18, weight="bold") # 達成粗體文字呈現
btn_color = _from_rgb((46, 80, 202)) 
bg_color = 'white'


label_le_cam = tk.Label(root, text="相機一(右)", font=boldFont, fg=btn_color, bg=bg_color)
label_le_cam.place(relx=0.75, rely=0, y=15, anchor='n')
left_canvas = tk.Canvas(root, bg='black')
left_canvas.config(width=640, height=360)
left_canvas.place(relx=0.02, y=60)

label_ri_cam  = tk.Label(root, text="相機二(左)", font=boldFont, fg=btn_color, bg=bg_color)
label_ri_cam.place(relx=0.25, rely=0, y=15, anchor='n')
right_canvas = tk.Canvas(root, bg='black')
right_canvas.config(width=640, height=360)
right_canvas.place(relx=0.52, y=60)

label_location = tk.Label(root, text="相機二(左)-座標值", font=boldFont, fg=btn_color, bg=bg_color)
label_location.place(relx=0.12, y=435, anchor='n')

label2_location = tk.Label(root, text="相機二(右)-座標值", font=boldFont, fg=btn_color, bg=bg_color)
label2_location.place(relx=0.4, y=435, anchor='n')

Left_listbox = tk.Listbox(root, height=12, width=35)
Left_listbox.place(relx=0.02, y=475)
Left_listbox.configure(background="skyblue4", foreground="white", font=('Aerial 13'))

Right_listbox = tk.Listbox(root, height=12, width=35)
Right_listbox.place(relx=0.30, y=475)
Right_listbox.configure(background="skyblue4", foreground="white", font=('Aerial 13'))

root.config(cursor="arrow")

val = 15
save_path = 'Photographed images/'
Pred_pix_loc = {}

btn = Button(root, text="開啟攝相機", command=lambda:[Begin(), video_loop()], fg='white', bg=btn_color, activebackground=btn_color, font=btn_boldFont)
btn.place(x=1025 , y=470, width=150, height=90)

btn = Button(root, text="儲存影像", command=lambda: [Stop(), take_snapshot(save_path)], fg='white', bg=btn_color, activebackground=btn_color, font=btn_boldFont)
btn.place(x=1200 ,y=470, width=150, height=90)


btn = Button(root, text="關閉UI", command=lambda: [Stop(), Close_UI(root)], fg='white', bg=btn_color, activebackground=btn_color, font=btn_boldFont)
btn.place(x=1200 ,y=590, width=150, height=90)



root.mainloop()

cv2.destroyAllWindows()
