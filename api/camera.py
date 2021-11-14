import cv2, queue, threading, time
from api.LED import LED
import glob, os
import imutils

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name, height=720, width=1280):
        self.cap = None 
        self.LED0 = LED()
        self.q = queue.Queue()
        try:
            print('init_height=', height)
            print('init_width=', width)
            
            self.cap = cv2.VideoCapture(name)
            self.cap.set(cv2.CAP_PROP_FPS, 5)
            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.setWidth(width)
            self.setHeight(height)
            

            # 取得影像的尺寸大小
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("Image Size: %d x %d" % (width, height))
            #self.cap.release()
            #self.cap = None
        except:
            print("camera connect error")

    def setWidth(self, width):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    def setHeight(self, height):
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def isOpened(self):
        return self.cap.isOpened()
        
    def open(self, name=0, height=720, width=1280):
        print('open().self.cap=', self.cap)
        if self.cap == None:
            Camera_path = glob.glob('/dev/vi*') # ('dev/video1')
            Camera_name = os.path.split(Camera_path[0])[1] # video1
            Camera_id = int(Camera_name[-1]) # 1 or 0 else
            self.cap = cv2.VideoCapture(Camera_id)
            self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
            self.setWidth(width)
            self.setHeight(height)
        time.sleep(3)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.0035)
        self.t2 = threading.Thread(target = self.LED0.LED_blink)
        self.t2.start()
        self.t1 = threading.Thread(target = self._reader)
        self.t1.daemon = True
        self.t1.start()

    def _reader(self):
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)


    def read(self):
        success = True
        if not self.q.empty():
            result = self.q.get()
        else:
            success = False
            result = False

        return success, result
    def release(self):
        self.t1.do_run = False
        self.t1.join()
        self.t2.do_run = False
        self.t2.join()
        #self.LED0.LED_close()
        self.cap.release()
        self.cap = None

