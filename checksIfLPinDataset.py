import csv
def check_if_licenseplate_in_dataset(checkPlateNumber):
    with open('licensePlatesNumbers.csv') as file:#opens a csv file
        csvreader = csv.reader(file)#reads from it
        header=[]
        header=next(csvreader)#gets headers
        print(header)
        licenseplates = []
        for row in csvreader:#gets all the cars license plates number
            licenseplates.append(row)
        print(licenseplates)
    for plateNumber in licenseplates:#check if the license plate checkPlateNumber is in the csv
        number=int(plateNumber[0])
        if(number==checkPlateNumber):#in-true , not in-false
            return True
    return False
