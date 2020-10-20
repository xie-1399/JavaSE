import os

#打包一下对应的txt文件，对应的生成的是文件名+类别
packages = os.listdir("/media/xxl/98D5D1E9544F330D/Python/Package_Recongise/train/")
with open("./dataset.txt","w") as f:
        count0=0
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        count6=0
        count7=0
        count8=0
        count9=0
        for package in packages:
            name = package.split(".")[0]
            name=str(name)
            if "bian" in name:
                f.write(package+";"+ "1\n")
                count1+=1
            elif "chong" in name:
                f.write(package+";"+"2\n")
                count2+=1
            elif "lun" in name:
                f.write(package+";"+"3\n")
                count3+=1
            elif "ruan" in name:
                f.write(package+";"+"4\n")
                count4+=1
            elif "shou" in name and "shoutibao" not in name:
                f.write(package+";"+"5\n")
                count5+=1
            elif "shoutibao" in name:
                f.write(package+";"+"6\n")
                count6+=1
            elif "yinger" in name:
                f.write(package+";"+"7\n")
                count7+=1
            elif "ying" in name and "yinger" not in name:
                f.write(package+";"+"8\n")
                count8+=1
            elif "zhixz" in name:
                f.write(package+";"+"9\n")
                count9+=1
            else:
                f.write(package+";"+"0\n")
                count0+=1
f.close()