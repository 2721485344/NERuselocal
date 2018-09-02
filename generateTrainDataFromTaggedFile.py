#encoding=utf8
import csv
rows=csv.reader(open("D:\\data\\taggeddata\\G45.901.csv",'r'))

rowCount=0
dev=open("example.dev",'w',encoding='utf8')
train=open("example.train",'w',encoding='utf8')
test=open("example.test",'w',encoding='utf8')
huanhang=['銆?,'!','锛?,'"','锛?,'?']
skip=['-']
flag=0
lenthTotal=0
devFlag = True
trainFlag = False
testFlag = False
lenthTotal=29247
lenthTotal2=34543
lenthTotal3=29906
biaoji = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW','CL']
for row in rows:
    if flag==0:
        flag=1
        continue
    if len(row)==6 and devFlag :
        rowCount+=1
        if rowCount<lenthTotal/15-1:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        else:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                devFlag=False
                testFlag=True
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n"+"\n")
    if len(row)==6 and testFlag :
        rowCount+=1
        if rowCount > lenthTotal / 15 and rowCount < 2 * lenthTotal / 15-1 :
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        if rowCount >= 2 * lenthTotal / 15-1 and testFlag:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                testFlag = False
                trainFlag=True
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
    if len(row)==6 and trainFlag:
        if row[2].strip() and row[2].strip() not in huanhang:
            for a in row[2].strip():
                if row[3].strip() == 'O':
                    train.write(a.strip() + " " + row[3].strip() + "\n")
                else:
                    if a == row[2].strip()[0]:
                        string = a.strip() + " " + "B-" + row[3].strip()
                        text = string.split()
                        assert len(text) == 2
                        train.write(a.strip() + " " + "B-" + row[3].strip() + "\n")
                    else:
                        string = a.strip() + " " + "I-" + row[3].strip()
                        text = string.split()
                        assert len(text) == 2
                        train.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
        else:
            string = row[2].strip() + " " + row[3].strip()
            text = string.split()
            if len(text) != 2:
                print(string)
            if row[2].strip() and row[3].strip():
                train.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")

flag=0
devFlag = True
trainFlag = False
testFlag = False
rowCount=0
for row in rows1:
    if flag==0:
        flag=1
        continue
    if len(row)==6 and devFlag :
        rowCount+=1
        if rowCount<lenthTotal/15-1:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        else:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                devFlag=False
                testFlag=True
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n"+"\n")
    if len(row)==6 and testFlag :
        rowCount+=1
        if rowCount > lenthTotal / 15 and rowCount < 2 * lenthTotal / 15-1 :
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        if rowCount >= 2 * lenthTotal / 15-1 and testFlag:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                testFlag = False
                trainFlag=True
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
    if len(row)==6 and trainFlag:
        if row[2].strip() and row[2].strip() not in huanhang:
            for a in row[2].strip():
                if row[3].strip() == 'O':
                    train.write(a.strip() + " " + row[3].strip() + "\n")
                else:
                    if a == row[2].strip()[0]:
                        string = a.strip() + " " + "B-" + row[3].strip()
                        text = string.split()
                        assert len(text) == 2
                        train.write(a.strip() + " " + "B-" + row[3].strip() + "\n")
                    else:
                        string = a.strip() + " " + "I-" + row[3].strip()
                        text = string.split()
                        assert len(text) == 2
                        train.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
        else:
            string = row[2].strip() + " " + row[3].strip()
            text = string.split()
            assert len(text) == 2
            if row[2].strip() and row[3].strip():
                train.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")


flag=0
devFlag = True
trainFlag = False
testFlag = False
rowCount=0
for row in rows2:
    if flag==0:
        flag=1
        continue
    if len(row)==6 and devFlag :
        rowCount+=1
        if rowCount<lenthTotal/15-1:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2

                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        else:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        dev.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            dev.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                devFlag=False
                testFlag=True
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    dev.write(row[2].strip() + " " + row[3].strip() + "\n"+"\n")
    if len(row)==6 and testFlag :
        rowCount+=1
        if rowCount > lenthTotal / 15 and rowCount < 2 * lenthTotal / 15-1 :
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+ row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:
                string = row[2].strip() + " " + row[3].strip()
                text = string.split()
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
        if rowCount >= 2 * lenthTotal / 15-1 and testFlag:
            if row[2].strip() and row[2].strip() not in huanhang:
                for a in row[2].strip():
                    if row[3].strip()=='O':
                        string = a.strip() + " " + row[3].strip()
                        text = string.split( )
                        assert len(text) == 2
                        test.write(a.strip()+" "+row[3].strip()+"\n")
                    else:
                        if a==row[2].strip()[0]:
                            string = a.strip() + " " + "B-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "B-"+row[3].strip()+ "\n")
                        else:
                            string = a.strip() + " " + "I-" + row[3].strip()
                            text = string.split( )
                            assert len(text) == 2
                            test.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
            else:

                testFlag = False
                trainFlag=True
                string = row[2].strip() + " " + row[3].strip()
                text = string.split( )
                assert len(text) == 2
                if row[2].strip() and row[3].strip():
                    test.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")
    if len(row)==6 and trainFlag:
        if row[2].strip() and row[2].strip() not in huanhang:
            for a in row[2].strip():
                if row[3].strip() == 'O':
                    string = a.strip() + " " + row[3].strip()
                    text = string.split( )
                    assert len(text) == 2
                    train.write(a.strip() + " " + row[3].strip() + "\n")
                else:
                    if a == row[2].strip()[0]:
                        string=a.strip() + " " + "B-" + row[3].strip()
                        text=string.split( )
                        assert  len(text)==2
                        train.write(a.strip() + " " + "B-" + row[3].strip() + "\n")
                    else:
                        string=a.strip() + " " + "I-" + row[3].strip()
                        text=string.split( )
                        assert  len(text)==2
                        train.write(a.strip() + " " + "I-" + row[3].strip() + "\n")
        else:
            string = row[2].strip() + " " + row[3].strip()
            text = string.split( )
            assert len(text) == 2
            if row[2].strip() and row[3].strip():
                train.write(row[2].strip() + " " + row[3].strip() + "\n" + "\n")

dev.close()
test.close()
train.close()
