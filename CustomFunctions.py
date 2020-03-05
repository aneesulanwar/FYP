import time, datetime,os, shutil
from PIL import Image
import io, base64, cv2
import re


def removeCharacter(word):


    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789 ./-')


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
                retDict["Name"] = dataSplit[0]
                adress = ""
                for i in range(1,len(dataSplit)):
                    adress += dataSplit[i]+" "
                retDict["Address"] = adress
            elif "ptcl_dateblock" in keys:
                dataSplit = data.split("\n")
                date = dataSplit[0].split(" ")
                if(len(date)>=2):
                    retDict["Date"] = date[-2]+" "+date[-1]
            elif "ptcl_grandtotal" in keys:
                dataSplit = data.split("\n")

                for amount in dataSplit:
                    if "Total" in amount or "Grand" in amount:
                        total = amount.split(" ")
                        retDict["Amount"] = total[-1]
            elif "ptcl_phonenumber" in keys:
                dataSplit = data.split("\n")
                retDict["PhoneNumber"] = dataSplit[-1]



        print(retDict)
        return retDict

    elif type == "0":
        retDict = {"Title": "GAS"}
        for data, keys in zip(list_values, list_keys):
            dataSplit = data.split("\n")
            print(dataSplit)
            if "gAddress" in keys:
                adress=""
                for data1 in dataSplit:
                    if "Name" in data1:
                        retDict["Name"] = data1.split(':')[1]
                    elif "Address" in data1:
                        adress += removeCharacter(data1.split(':')[1])
                    else:
                        adress += removeCharacter(data1)
                retDict["Address"] = adress
            elif "gData2" in keys:
                retDict["Date"] = removeCharacter(dataSplit[0])
                retDict["Amount"] = removeCharacter(dataSplit[1])
            elif "gMeter" in keys:
                retDict["Meter"] = dataSplit[0]
            elif "gUnit" in keys:
                for k in range(0,len(dataSplit)-1):
                    if "Diff" in dataSplit[k]:
                        retDict["Units"] = dataSplit[k+1]

        print(retDict)
        return retDict





# Creates a single sub-directory
# def createNewDir2():
#     timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
#     folderpath = os.path.join( "OutputsHistory", str(timenow))
#     if not os.path.exists(folderpath):
#         os.makedirs(folderpath)
#         print ('Created:', folderpath)
