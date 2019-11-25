#program 1 FindS
f = open('prg1.csv','r')
lst = f.readline().split(',')       
length=len(lst)
f.close();
f = open('prg1.csv','r')
count=1
hypo=['0']*(length-1)
print("Intial Hypothesis is = ",hypo)
for value in f:
    lst = value.split(',')
    if(lst[-1] == "yes\n"):
        for i in range(0, length-1):
            if(hypo[i]!=lst[i] and hypo[i]!='0'):
                hypo[i]='?'
            else:
                hypo[i]=lst[i]   
    print("Hypothesis after row ", count ," = ",hypo )
    count=count+1;
f.close()
