import socket
from turtle import width
from PIL import Image
from tkinter import *
#import projectGUI
#import projectServer
#projectServer.server.
class Client:
    def __init__(self,ip='127.0.0.1',port=12345):
        self.clientSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect((ip,port))
    def send_message(self,msg):
        self.clientSocket.send(str(msg).encode())
    def receive_message(self,):
        data=self.clientSocket.recv(1024).decode()
        return data
    def send_image(self,):
        filePath='vehicle1.jpg'
        img = Image.open(filePath)
        width, height = img.size
        imgSize=width*height
        self.send_message(str(imgSize))
        with open(filePath,'rb') as f:
            while True:
                data=f.read(1024)
                if not data:
                    print('{0} send over...'.format(filePath))
                    break
                self.clientSocket.send(data)
    def exit(self,):
        self.clientSocket.close()
#pops a window with a color depending if licenseplate in csv green-in| red-not in
def pop_window(in_dataset):
    width=500
    height=500
    if(in_dataset):
        color='green'
        text='license is in the system'
        gui = Tk(className='confirmatoin')
        # set window size
        gui.geometry(f"{width}x{height}")
        #set window color
        gui.configure(bg=color)
        label1 = Label(gui, text=text, fg='black', bg=color,font=("Courier", 15)).place(x=(width/2)-(8*len(text)),y=height/2-50)
        gui.mainloop() 
    else:
        color='red'
        text='license is not in the system'
        gui = Tk(className='confirmatoin')
        # set window size
        gui.geometry(f"{width}x{height}")
        #set window color
        gui.configure(bg=color)
        label1 = Label(gui, text=text, fg='black', bg=color,font=("Courier", 15)).place(x=(width/2)-(7*len(text)),y=height/2-50)
        gui.mainloop() 
pop_window(False)

if __name__ == '__main__':
    client=Client()
    data=client.receive_message()
    print(data)
    client.send_image()











