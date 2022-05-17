import os
import socket
from licensePlateDetection import plate_number
class Server:
    def __init__(self,ip='0.0.0.0',port=12345):
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.bind((ip,port))
        self.serverSocket.listen(1)
        print("Wait for Connection.....................")
    def connect_to_client(self,):
        self.clientSocket,self.clientAddres=self.serverSocket.accept()
        print(f"connected to {self.clientAddres}")
    def send_message(self,msg):
        self.clientSocket.send(str(msg).encode())
    def receive_message(self,):
        data=self.clientSocket.recv(1024).decode()
        return data
    def receive_image(self,):
        filePath='vehicle2.jpg'
        with open(filePath,'wb') as f:
            #add receive image size
            receivedSize=int(self.receive_message())
            currentSize=0
            while not receivedSize==currentSize:
                if receivedSize-currentSize>1024:
                    data=self.clientSocket.recv(1024)
                    currentSize+=len(data)
                else:
                    data=self.clientSocket.recv(1024)
                    currentSize=receivedSize
                f.write(data)
    def exit(self,):
        self.clientSocket.close()

if __name__ == '__main__':
    server=Server()
    while True:
        server.connect_to_client()
        server.send_message("connected to server")
        server.receive_image()

    