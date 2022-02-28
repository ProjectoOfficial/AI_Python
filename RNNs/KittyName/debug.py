l = list()
import io
path = "/home/daniel/PycharmProjects/gpuproject/KittyName/names.txt"
path2 = "/home/daniel/PycharmProjects/gpuproject/KittyName/names2.txt"
prohibited = ["(",")","\"","'","|","$","%","&","/","=","?","^","*","]","[","\t",".",",",";",":","#","@","-","_","§","°","\\","~","`"]
with open(path, 'r') as f:
    lines = f.readlines()
    for s in lines:
        s = s.lower()
        s = ''.join([i for i in s if not i.isdigit()
                     and not i.isspace()
                     and i not in prohibited])
        print(s)
        l.append(s)
        f.close()

with open(path2,'w') as f2:
    l2 =sorted(l)
    for line in l2:
        li = line+"\n"
        f2.writelines(li)
    f2.close()