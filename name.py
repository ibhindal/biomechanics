file_name=[]

for x in range(1, 3, 1 ):
 for y in range(30, 60, 10):
    for z in range(1, 4, 1 ):
        name =("P2_" + str(x) +"_" +str(y) +"_" + str(z)+ ".mp4")
        print (name)
        file_name.append(name) 

#print (file_name)

i=0
file_name_len=len(file_name)
for i in range(file_name_len): 
    name = file_name[i]
print (name)
    
