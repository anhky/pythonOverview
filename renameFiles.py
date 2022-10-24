import os
import cv2
#  Path of current directory:
path = os.getcwd()
files = os.listdir(path)
i = 1
for file in files:
    try:
        file_name, file_extension = os.path.splitext(os.path.basename(file))
        if file_extension !='.py' and file_extension !='.png':
            os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.png'))
            i = i+1
            cv2.imwrite(file, i)
        else:
            pass
    except:
        pass
