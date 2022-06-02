import sys
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from PIL import ImageTk,Image
import projectClient



class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.canvas.place(anchor=tkinter.NW)
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=90, command=self.snapshot)
        self.btn_snapshot.pack()
        self.btn_snapshot.place(anchor=tkinter.NW,x=0,y=self.vid.height)
        # canvas for ractangle color
        self.confirmationBox=tkinter.Canvas(width = int(self.vid.width)*2, height = 100,background="gray")
        self.confirmationBox.pack()
        self.confirmationBox.place(x=0,y=505)
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            print(frame)
            cv2.imwrite("vehicle1" +".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.show_snapshot()
    def show_snapshot(self):
        self.imgSnapshot = ImageTk.PhotoImage(Image.open("vehicle1.jpg"))
        snapshot=tkinter.Label(self.window,image=self.imgSnapshot)
        snapshot.pack()
        snapshot.place(anchor = tkinter.NW,x=int(self.vid.width))
        # Button to start the program
        self.btn_run_program=tkinter.Button(self.window, text="start", width=90, command=self.start_program)
        self.btn_run_program.pack()
        self.btn_run_program.place(anchor=tkinter.NW,x=self.vid.width,y=self.vid.height)
    def start_program(self):
        isInDataset=projectClient.main()
        print(isInDataset)
        self.show_if_LP_in_dataset(isInDataset)
    def show_if_LP_in_dataset(self,isInDataset):
        if(isInDataset=="found"):
            self.confirmationBox.config(background="green")
        elif(isInDataset=="not found"):
            self.confirmationBox.config(background="red")
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)
    
 
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(self.height)
        print(self.width)

    def get_frame(self):
        if self.vid.isOpened:
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        sys.exit()

def main():
    window = tkinter.Tk()
    window.geometry("1285x610")
    window.configure(background='white')
    App(window, "video camera feed")
main()