import csv
def check_if_licenseplate_in_dataset(checkPlateNumber):
    with open('licensePlatesNumbers.csv') as file:
        csvreader = csv.reader(file)
        header=[]
        header=next(csvreader)
        print(header)
        licenseplates = []
        for row in csvreader:
            licenseplates.append(row)
        print(licenseplates)
    for plateNumber in licenseplates:
        number=int(plateNumber[0])
        if(number==checkPlateNumber):
            return True
    return False

isIn=check_if_licenseplate_in_dataset(2952165)
print(isIn)
