from tkinter import *
import socket
import os
try:
    from PIL import Image
except ImportError:
    import Image


class Client:
    def __init__(self,ip='127.0.0.1',port=12345):
        self.clientSocket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)#create a socket
        self.clientSocket.connect((ip,port))#connect to server with ip port
    def send_message(self,msg):
        self.clientSocket.send(str(msg).encode())#send encoded masseges to server
    def receive_message(self,):
        data=self.clientSocket.recv(1024).decode()#receive encoded masseges from server decodes it and return data
        return data
    def send_image(self,):
        filePath='vehicle1.jpg'
        imgSize=os.path.getsize(filePath)
        self.send_message(str(imgSize))#sends the img size to client
        with open(filePath,'rb') as f:#open file and reads binary from it
            while True:
                data=f.read(1024)#reads data
                if not data:
                    print('{0} send over...'.format(filePath))
                    break
                self.clientSocket.send(data)#sends data
    def exit(self,):
        self.clientSocket.close()

def main():
    client=Client()
    data=client.receive_message()
    print(data)
    client.send_image()
    isInDataset=client.receive_message()
    print(isInDataset)
    client.exit()
    return isInDataset










