from concurrent.futures import thread
import os
import socket
from licensePlateDetection import GetLicensePlateNumber
from checksIfLPinDataset import check_if_licenseplate_in_dataset
import threading
class Server:
    def __init__(self,ip='0.0.0.0',port=12345):
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #create a socket
        self.serverSocket.bind((ip,port))#bind it with ip and port
        self.serverSocket.listen(1)#listen to any connections
        print("Wait for Connection.....................")
    def connect_to_client(self,):
        self.clientSocket,self.clientAddres=self.serverSocket.accept()#accept connection to server
        print(f"connected to {self.clientAddres}")
    def send_message(self,msg):
        self.clientSocket.send(str(msg).encode())#send encoded masseges to client
    def receive_message(self,):
        data=self.clientSocket.recv(1024).decode()#receive encoded masseges from client decodes it and return data
        return data
    def receive_image(self,):
        filePath='vehicle2.jpg'
        with open(filePath,'wb') as f: #open new file to write to
            receivedSize=int(self.receive_message())#gets image size from client
            print(receivedSize)
            currentSize=0
            #receivce image data until receivedSize==currentSize
            while not receivedSize==currentSize:
                if receivedSize-currentSize>1024:
                    data=self.clientSocket.recv(1024)
                    currentSize+=len(data)
                else:
                    data=self.clientSocket.recv(1024)
                    currentSize=receivedSize
            print('received image')
    def exit(self,):
        self.clientSocket.close()#closes connection
        self.serverSocket.close()

def get_licenseplate_number():
    licenseplateNumber=GetLicensePlateNumber()
    licenseplateNumber.img_processing()
    licenseplateNumber.find_contours()
    result = licenseplateNumber.straighten_licenseplate()
    char=licenseplateNumber.segment_characters(result)
    licenseplateNumber.get_img_numbers(char)
    licenseplateNumber.load_saved_weights()
    plateNumber=licenseplateNumber.predict_numbers_value()
    print(plateNumber)
    if(plateNumber==""):
        plateNumber=0
    return plateNumber
def is_LP_in_dataset(plateNumber):
    isIn=check_if_licenseplate_in_dataset(plateNumber)
    return isIn

def establish_connection():
    server=Server()
    server.connect_to_client()
    thread = threading.Thread(target=main,args=(server,))
    thread.start()

def main(server):
    while True:
        server.send_message("connected to server")
        server.receive_image()
        plateNumber=int(get_licenseplate_number())
        print(plateNumber)
        isInDataset=is_LP_in_dataset(plateNumber)
        print(isInDataset)
        if(isInDataset):
            server.send_message("found")
        else:
            server.send_message("not found")
        server.exit()
        establish_connection()
establish_connection()