import pandas as pd
import numpy as np

dataset=pd.read_csv('PlayTennis.csv')
tdata=dataset.sample(frac=0.8)
print("The Training Dataset:\n")
print(dataset)

def ID3(data,f,tname,pnode):
    if len(np.unique(data[tname]))==1:
        return np.unique(data[tname])[0]
    elif len(f)==0:
        return pnode
    else:
        pnode=np.unique(data[tname])[np.argmax(np.unique(data[tname])[1])]
        item_values=[InfoGain(data,feature,tname)for feature in f]
        bfi=np.argmax(item_values)
        bf=f[bfi]
        tree={bf:{}}
        f=[i for i in f if i!=bf]
        for value in np.unique(data[bf]):
            sub_data=data.where(data[bf]==value).dropna()
            subtree=ID3(sub_data,f,tname,pnode)
            tree[bf][value]=subtree
        return(tree)
    
def InfoGain(data,feature,tname):
    te=entropy(data[tname])
    vals,counts=np.unique(data[feature],return_counts=True)
    we=np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[feature]==vals[i]).dropna()[tname])for i in range(len(vals))])
    Information_Gain=te-we
    return Information_Gain

def entropy(target_col):
    elements,counts=np.unique(target_col,return_counts=True)
    entropy=np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

f=['Outlook','Temperature','Humidity','Wind']
tname="PlayTennis"
pnode=None
tree=ID3(dataset,f,tname,pnode)
print("\nTree:\n")
print(tree)
query=dataset.iloc[:,:].to_dict(orient="records")

def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result=tree[key][query[key]]
            except:
                return default
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


print("\nQuery:\n")
print(query[10])
result=predict(query[10],tree,1.0)
print("\n\nTesting sample 1:\n")
print(query[10],"PREDICTED=>",result)
result=predict(query[12],tree,1.0)
print("\nTesting sample 2:")
print(query[12],"PREDICTED=>",result)
