from turtle import update
import pandas as pd
import numpy as np
import csv
from tkinter import *
from pandastable import Table, TableModel,config
import time

class DatasetApp:
    def __init__(self,root):
        self.root=root
        self.highet=400
        self.file='licensePlatesNumbers.csv'
        self.frame = Frame(self.root)
        self.frame.pack(fill=BOTH,expand=True)
        self.frame.place(anchor=N,x=350,width=700)
    def show_table(self):
        df = pd.read_csv(self.file)
        self.pt = Table(self.frame, dataframe=df)
        self.pt.show()
    def update_table(self):
        df = pd.read_csv(self.file)
        self.pt.model.df=df
        self.pt.redraw()
    def add_licenseplate_to_dataset(self):
        # Use .get() to get the values of the StringVars, not the names of the StringVars
        def info():
            list_of_lists = [f'{licensePlateNumber.get()}',f'{carBrand.get()}',f'{carColor.get()}',]
            print(list_of_lists)
            licensePlateNumber_entry.delete(0, 'end')
            carBrand_entry.delete(0, 'end')
            carColor_entry.delete(0, 'end')
            with open(self.file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list_of_lists)
            self.update_table()

        # Add root to put the all the objects in the same GUI
        titel_text = Label(self.root, text = "Add License Plate",font=("Arial", 15))
        entry1_text = Label(self.root, text = "Type license plate number * ",)
        entry2_text = Label(self.root, text = "Type car brand * ",)
        entry3_text = Label(self.root, text = "Type car color * ",)
        
        titel_text.place(x = 80, y = self.highet-50)
        entry1_text.place(x = 80, y = 30+self.highet)
        entry2_text.place(x = 80, y = 90+self.highet)
        entry3_text.place(x = 80, y = 150+self.highet)

        licensePlateNumber = StringVar()
        carBrand = StringVar()
        carColor = StringVar()

        # Add root to put the all the objects in the same GUI
        licensePlateNumber_entry = Entry(self.root, textvariable = licensePlateNumber, width = "10")
        carBrand_entry = Entry(self.root, textvariable = carBrand, width = "10")
        carColor_entry = Entry(self.root, textvariable = carColor, width = "10")

        licensePlateNumber_entry.place(x = 80, y = 60+self.highet)
        carBrand_entry.place(x = 80, y = 120+self.highet)
        carColor_entry.place(x = 80, y = 180+self.highet)

        addLicensePlateBtn = Button(self.root,text = "Add", width = "10", height = "2", command = info, bg = "lightgreen")
        addLicensePlateBtn.place(x = 80, y = 240+self.highet)

    def remove_licenseplate_from_dataset(self):
        # Use .get() to get the values of the StringVars, not the names of the StringVars
        def info():
            input = int(f'{licensePlateNumber.get()}')
            print(input)
            licensePlateNumber_entry.delete(0, 'end')
            df = pd.read_csv(self.file)
            print(df)
            df =  df[df.plateNumber != input] 
            print(df)
            # df.column_name != whole string from the cell
            # now, all the rows with the column: Name and Value: "dog" will be deleted
            df.to_csv(self.file, index=False)
            self.update_table()
        # Add root to put the all the objects in the same GUI
        titel_text = Label(self.root, text = "Remove License Plate",font=("Arial", 15))
        entry1_text = Label(self.root, text = "Type license plate number * ",)

        titel_text.place(x = 400, y = self.highet-50)
        entry1_text.place(x = 400, y = 30+self.highet)

        licensePlateNumber = StringVar()


        # Add root to put the all the objects in the same GUI
        licensePlateNumber_entry = Entry(self.root, textvariable = licensePlateNumber, width = "10")
        licensePlateNumber_entry.place(x = 400, y = 60+self.highet)

        removeLicensePlateBtn = Button(self.root,text = "Remove", width = "10", height = "2", command = info, bg = "lightgreen")
        removeLicensePlateBtn.place(x = 400, y = 240+self.highet)

#remove_licenseplate_from_dataset()
#add_licenseplate_to_dataset()
def main():
    root = Tk()
    root.geometry('700x700')
    root.title("Manage Dataset")
    app=DatasetApp(root)
    app.show_table()
    app.add_licenseplate_to_dataset()
    app.remove_licenseplate_from_dataset()
    root.mainloop()
main()


