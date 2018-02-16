#!/usr/bin/python3

from subprocess import Popen, PIPE
DIRECTORY= '/home/mkim/cuda-uflic/build/release/'
HEIGHT = [256, 512, 1024, 2048]
WIDTH = [512,1024,2048,4096]
EXECNAME = 'UFLIC_rendering'
F = open('output', 'w')

def convertImage(f,w=None,h = None):
    remaining = f
    if w != None:
         remaining = remaining +'_'+ str(w)
    if h != None:
        remaining = remaining + '_' + str(h)
    print(['mv', 'uflic-final.png', 'uflic-final_' + remaining + '.png'])
    p = Popen(['convert', 'uflic-final.pnm', 'uflic-final_' + remaining + '.png'])
    #p = Popen(['mv', 'uflic-final.png', 'uflic-final_' + remaining + '.png'], stdout=PIPE, stderr=PIPE)

FILETYPE = ['bfield', 'PSI']
for f in FILETYPE:
    execname = DIRECTORY+EXECNAME
    print([execname, f])
    F.writelines(f)
    p = Popen([execname, f], stdout=PIPE, stderr=PIPE)
    p.wait()
    out, err = p.communicate()
    print(out.decode("utf-8"))
    F.writelines(out.decode("utf-8"))
    convertImage(f)

execname = DIRECTORY+EXECNAME
for w,h in zip(WIDTH,HEIGHT):
    F.writelines([execname, "dims",str(w),str(h) + '\n'])
    print([execname, "dims",str(w),str(h)])
    p = Popen([execname, "dims", str(w),str(h)], stdout=PIPE, stderr=PIPE)
    p.wait()
    out, err = p.communicate()
    print(out.decode("utf-8"))
    F.writelines(out.decode("utf-8"))
    convertImage('doublegyre',w,h)

