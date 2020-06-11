import time, datetime,os, shutil
from PIL import Image
import io, base64, cv2
import re


def removeCharacter(word):


    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 ./-')


    answer = ''.join(filter(whitelist.__contains__, word))

    #result = re.sub('[\W_]+', '', word)

    # printing final string
    return answer



def listToString(lst):
    # initialize an empty string
    resStr = ""
    # traverse in the string
    for node in lst:
        resStr += node
    return resStr

# Recursively creates sub directory based on current time
def createNewDir(destFolder):
    # os.makedirs("/OutputsHistory" + time.strftime("/%Y/%m/%d", time.gmtime(time.time()-3600)))
    mydir = os.path.join(destFolder, datetime.datetime.now().strftime('%Y/%m/%d_%H-%M-%S'))
    if not os.path.exists(mydir):
        os.makedirs(mydir)
        print('Created New Dir:', mydir)
    return mydir

# Moves Files from one driectory to another
def moveFilesToNewDir(src,dest):
    src_folder = os.listdir(src)
    for fileT in src_folder:
        srcname = os.path.join(src, fileT)
        dstname = os.path.join(dest, fileT)
        shutil.move(srcname, dstname)
    print('Transfered Output Files To: ', dest)

    return 1


# Convert Image to base 64 string
def getImageBytes(filePath):
    with open(filePath, "rb") as imageFile:
        base64_str = base64.b64encode(imageFile.read())
        # print(base64_str)

    return str(base64_str)




def processText(recdict, type):
    list_values = [v for v in recdict.values()]
    list_keys = [ k for k in recdict]
    print(list_values)
    print(list_keys)

    if type == "2":
        retDict = {"Title": "PTCL"}
        for data,keys in zip(list_values,list_keys):

            if "ptcl_address" in keys:
                dataSplit = data.split("\n")
                retDict["Name"] = removeCharacter(dataSplit[0])
                adress = ""
                for i in range(1,len(dataSplit)):
                    adress += removeCharacter(dataSplit[i])+" "
                retDict["Address"] = adress
            elif "ptcl_dateblock" in keys:
                dataSplit = data.split("\n")
                date = dataSplit[0].split(" ")
                if(len(date)>=2):
                    retDict["Date"] = removeCharacter(date[-2]+" "+date[-1])
            elif "ptcl_grandtotal" in keys:
                dataSplit = data.split("\n")

                for amount in dataSplit:
                    if "Total" in amount or "Grand" in amount:
                        total = amount.split(" ")
                        amountParse = removeCharacter(total[-1])
                        retDict["Amount"] = amountParse
            elif "ptcl_phonenumber" in keys:
                dataSplit = data.split("\n")
                retDict["PhoneNumber"] = removeCharacter(dataSplit[-1])



        print(retDict)
        return retDict

    elif type == "0":
        retDict = {"Title": "SUI GAS"}
        for data, keys in zip(list_values, list_keys):
            dataSplit = data.split("\n")
            print(dataSplit)
            if "gAddress" in keys:
                adress = ""
                count = 0
                for data1 in dataSplit:
                    if "Name" in data1 or "Naine" in data1:
                        word = removeCharacter(data1.split(':')[1])
                        if word != "":
                            count += 1
                        retDict["Name"] = word

                    elif "Address" in data1:
                        words = data1.split(':')
                        if len(words) >= 3:
                            word = removeCharacter(words[2])
                        else:
                            word = ''
                        adress += word+" "
                    else:
                        word = removeCharacter(data1)
                        if "Bill" in word or "Account" in word or "Consumer" in word or " " == word or "" == word:
                            word = ""
                        else:
                            if count == 0:
                                retDict["Name"] = word
                                count +=1
                            else:
                                adress += word+" "
                retDict["Address"] = adress
            elif "gData" in keys:
                dateCond = False
                amountCond = False
                for k in dataSplit:
                    word = removeCharacter(k)
                    if word.strip():
                        print ("word is ", word)
                        if dateCond == False:
                            retDict["Date"] = word
                            dateCond = True
                        elif amountCond == False:
                            retDict["Amount"] = word
                            amountCond = True


                if len(dataSplit)>0:
                    dateCond = True
                else:
                    retDict["Data"] = "Sep 2016"
                if len(dataSplit) > 1:
                    dateCond = True
                else:
                    retDict["Amount"] = "0"

            elif "gMeter" in keys:
                find = False
                for k in range(0,len(dataSplit)-1):
                    if "Meter" in dataSplit[k]:
                        retDict["Meter"] = removeCharacter(dataSplit[k+1])
                        find = True
                if not find:
                    for k in range(0, len(dataSplit)):
                        word = removeCharacter(dataSplit[k])
                        if word !='':
                            retDict["Meter"] = word
                            break

            elif "gUnit" in keys:
                for k in range(0,len(dataSplit)-1):
                    if "Diff" in dataSplit[k]:
                        if dataSplit[k+1] != "" or dataSplit[k+1] != " ":
                            retDict["Units"] = removeCharacter(dataSplit[k+1])
                        else:
                            retDict["Units"] = removeCharacter(dataSplit[k+2])

        print(retDict)
        return retDict

    elif type =="1":


        retDict = {"Title": "IESCO"}
        for data, keys in zip(list_values, list_keys):

            if "NAME&ADDRESS" in data or "NAME & ADDRESS" in data or "NAME" in data or "ADDRESS" in data:
                print("True Name")
                dataSplit = data.split("\n")
                if len(dataSplit) > 0:
                    retDict["Name"] = removeCharacter(dataSplit[1]) + removeCharacter(dataSplit[2])
                elif len(dataSplit) == 0:
                    retDict["Name"] = "Name"
                if len(dataSplit) > 3:
                    addr = ""
                    for i in range(3,len(dataSplit)):
                        if "W/O" not in dataSplit[i] and "S/O" not in dataSplit[i] and "D/O" not in dataSplit[i]:
                            addr += removeCharacter(dataSplit[i])

                    retDict["Address"] = addr
                elif len(dataSplit) == 1:
                    retDict["Address"] = "Address"

            elif "W/O" in data or "S/O" in data or "D/O" in data or "H.NO" in data:
                dataSplit = data.split("\n")

                if (len(dataSplit) <= 6):

                    if len(dataSplit) > 0:
                        retDict["Name"] = removeCharacter(dataSplit[0])
                    elif len(dataSplit) == 0:
                        retDict["Name"] = "Name"
                    if len(dataSplit) > 3:
                        retDict["Address"] = removeCharacter(dataSplit[2]) + removeCharacter(dataSplit[3])
                    elif len(dataSplit) == 1 or len(dataSplit) == 0:
                        retDict["Address"] = "Address"
                    elif len(dataSplit) == 3:
                        retDict["Address"] = removeCharacter(dataSplit[2])

                else:

                    if len(dataSplit) > 0:
                        retDict["Name"] = removeCharacter(dataSplit[1]) + removeCharacter(dataSplit[2])

                    elif len(dataSplit) == 0:
                        retDict["Name"] = "Name"

                    if len(dataSplit) > 3:
                        addr = ""
                        for i in range(3, len(dataSplit)):
                            if "W/O" not in dataSplit[i] and "S/O" not in dataSplit[i] and "D/O" not in dataSplit[i]:
                                addr += removeCharacter(dataSplit[i])

                        retDict["Address"] = addr
                    elif len(dataSplit) == 1:
                        retDict["Address"] = "Address"


            elif "BILL MONTH" in data or "BILL" in data or "MONTH" in data:

                dataSplit = data.split("\n")
                retDict["Date"] = removeCharacter(dataSplit[len(dataSplit)-1])

            elif "METER" in data:

                dataSplit = data.split("\n")
                if len(dataSplit) == 2:
                    retDict["Meter"] = removeCharacter(dataSplit[1])
                elif len(dataSplit) == 3:
                    retDict["Meter"] = removeCharacter(dataSplit[2])

            elif ("PAYABLE" in data or "AYABLE" in data or "DUE" in data):
                dataSplit = data.split("\n")
                found = False
                for i in range(0,len(dataSplit)):
                    if "PAYABLE" in dataSplit[i] or "AYABLE" in dataSplit[i] or "DUE" in dataSplit[i]:
                        found = True
                        if i < len(dataSplit)-1:
                            i += 1
                    if found and (dataSplit[i] != "" or dataSplit[i] != " ") and len(dataSplit[i]) > 5:
                        amdict = dataSplit[i].split(" ")
                        retDict["Amount"] = removeCharacter(amdict[len(amdict)-1])

                        break
            elif "UNITS" in data or "CONSUM" in data:
                dataSplit = data.split("\n")
                retDict["Unit"] = removeCharacter(dataSplit[len(dataSplit)-1])
        print(retDict)
        return retDict






def removeLine(file):

    print("Funcrions Called", file)
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    #result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('detected_lines', detected_lines)
    # cv2.imshow('image', image)
    # cv2.imshow('result', result)
    cv2.imwrite(file, image)

# Creates a single sub-directory
# def createNewDir2():
#     timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
#     folderpath = os.path.join( "OutputsHistory", str(timenow))
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath)
#         print ('Created:', folderpath)
