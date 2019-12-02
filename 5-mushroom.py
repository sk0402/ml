import pandas as pd
mush = pd.read_csv('mushrooms.csv')
target = 'class'
classes = mush[target].unique()
features = mush.columns[mush.columns!=target]
testData = mush.sample(frac=0.3)
mush.drop(testData.index,inplace = True)
first ={}
fourth ={}
for x in classes:
    mushcl = mush[mush[target]==x][features]
    tot = len(mushcl)
    second={}
    for col in mushcl.columns:
        third={}
        for val,cnt in mushcl[col].value_counts().iteritems():
            prob = cnt/tot
            third[val]=prob
            second[col]=third
    first[x]=second
    fourth[x]=len(mushcl)/len(mush)
def proabs(params):
    proab={}
    for x in classes:
        calc = fourth[x]
        for col, val in params.iteritems():
            try:
                calc = first[x][col][val]
            except KeyError:
                calc =0
        proab[x]=calc
    return proab
def maxx(params):
    proab = proabs(params)
    maxcl =''; maxv=0
    for col,val in proab.items():
        if(val>maxv):
            maxv=val
            maxcl=col
    return maxcl

b=[]
for i in mush.index:
    b.append(   maxx(mush.loc[i,features]) == mush.loc[i,target]
            )
print(sum(b),'correct of',len(b))
print('Accuracy =',sum(b)/len(b))
b=[]
for i in testData.index:
    b.append(   maxx(testData.loc[i,features]) == testData.loc[i,target]
            )
print(sum(b),'correct of',len(b))
print('Accuracy =',sum(b)/len(b))
