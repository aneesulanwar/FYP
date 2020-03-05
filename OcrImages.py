from PreProcessImage import pre_process_image

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
import glob
import cv2
import PreProcessImage
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

def ocr_images():
    img_dir = 'Outputs/' # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files=glob.glob('Outputs'+'/*.*')
    textArr=[]
    textDict={}
    countFiles=0
    for f1 in files:
        if 'androidFlask' not in f1 and 'cropped_gData' not in f1 and 'gData0' not in f1 and 'gData1' not in f1 and ".txt" not in f1:
            # Pre-Process Image first
            #if "gData" not in f1:
            #pre_process_image(f1)

            # Read Image
            #img = cv2.imread(f1)
            if "gData" not in f1 and "ptcl" not in f1:
                l=1
                #PreProcessImage.set_image_dpi(f1)
                #PreProcessImage.skew_corretion(f1)
                #PreProcessImage.remove_noise_and_smooth(f1)
            if "ptcl" in f1:
                pre_process_image(f1)
            print("File name", f1)
            img = cv2.imread(f1)
            # Convert to Grey scale
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
            # text = pytesseract.image_to_string(Image.open('Outputs/cropped_ptcl_address 98.16%.png'))

            text = pytesseract.image_to_string(img)
            textArr.append(text)
            #print(text)

            #below commented line makes the keys of result as integer
            #key of dict here becomes an incrementing integer rather than a string
            #textDict[countFiles]= text

            #below code makes keys of result as filename string
            # and removes everything after a space occurs in string
            filename=os.path.basename(f1)
            sep = ' '
            newFilename = filename.split(sep, 1)[0]
            if newFilename in textDict:
                newFilename+="1"
            textDict[os.path.basename(newFilename)]= text


            countFiles+=1

    with open('Outputs/ocr_output.txt', 'w') as f:
        for item in textArr:
            f.write("%s\n" % item)

    return textDict