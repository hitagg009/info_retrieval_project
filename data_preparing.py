import csv
my_dict={}
rate=[]
ans=0
with open('iclr2017_papers.csv', 'r',encoding="utf-8") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        t=0;
        r=0;
        for i in row:
            t=t+1;
            r=r+1;
            ans=max(ans,t)
            if(t==6):
                my_dict[i]=""
            if(r==10):
                if(i=="decision"):
                    rate.append(i)
                if(i=="Accept (Oral)"):
                    rate.append("1")
                if(i=="Accept (Poster)"):
                    rate.append("2")
                if(i=="Invite to Workshop Track"):
                    rate.append("3")
                if(i=="Reject"):
                    rate.append("4")
csvFile.close()
print("length is")
print(ans)
with open('iclr2017_conversations.csv','r',encoding="utf-8") as csvfile:
    reader1=csv.reader(csvfile)
    k=0;
    for row1 in reader1:
        t=0;
        title=""
        for i in row1:
            t=t+1;
            if(t==6):
                title=i;
            if(t==10):
                my_dict[title]+=i;
for i in my_dict:
    s=""
    for j in my_dict[i]:
        if(j!="\n" and j!="\t"):
            s+=j;
        my_dict[i]=s;
my_list=[["class","Title","Content"]]
k=0
y=-1
for i in my_dict.keys():
    k=k+1
    y=y+1
    temp=[]
    temp.append(rate[y])
    temp.append(i)
    temp.append(my_dict[i])
    if(k!=1):
        my_list.append(temp)
print(k)
with open('datasetfinaluse.csv','w',encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(my_list)
print("done")