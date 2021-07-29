

import os

folder_path = "../../results/"

for dirpath, dirnames, files in os.walk(folder_path):
    if files:
        for f in files:
            os.remove(os.path.join(dirpath,f))
    for dir in dirnames:
        for dp, dn, fi in os.walk(dir):
            if fi:
                for f in files:
                    os.remove(os.path.join(dp,f))