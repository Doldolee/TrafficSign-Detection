from cgi import test
import json
from collections import Counter
from xml.etree.ElementTree import Element,SubElement,ElementTree,dump

# Check Class Frequency
## input: class name /  output: frequency on test & train dataset
## check the class that is more than 100
def check_class_freq(file_name, class_name, check_target_class=False):
    all_class=[]
    train_freq=0
    test_freq=0
    with open(f"../tt100k_2021/{file_name}.json") as json_file:
        anno_data = json.load(json_file)

        for i in anno_data['imgs']:
            data = anno_data['imgs'][i]
            
            split_other_alp = list(data['path'])[0]
            split_train_test_alp = list(data['path'])[1]
            #other dataset
            if split_other_alp =='o':
                continue
            # train, test dataset
            for j in data['objects']:
                label = j['category']
                all_class.append(label)
                #count class freq
                if split_train_test_alp=="r" and label==class_name:
                    train_freq +=1
                elif split_train_test_alp=="e" and label==class_name:
                    test_freq +=1

        print(f"{class_name} train freq = " , train_freq, f"{class_name} test freq = ", test_freq )    
        print(f"{class_name} all freq = ", train_freq + test_freq)
        if check_target_class:
            count = dict(Counter(all_class).most_common())
            print(count)
        
    return train_freq, test_freq
### check_class_freq("annotations_all","pne", True)


# target class name
def target_class_name():
    with open("./class_line.txt","r") as class_name:
        class_name = class_name.readline().strip().split(",")
        return class_name

# ids(train or val)
def dataset_ids(set):
    with open(f"../Data/{set}_ids_TT.txt","r") as ids:
        data_ids=[]
        while True:
            line = ids.readline()
            if not line: break
            line = line.strip()
            data_ids.append(line)
        return data_ids

# separate the annotation by image and save it in txt format
def json2txt():
    with open("../tt100k_2021/annotations_all.json","r") as json_data:
        json_data = json.load(json_data)
    total = len(list(json_data['imgs']))
    all_file_name=[]
    for i in range(total):
        core_data = list(json_data['imgs'].values())[i]
        #throw away other category data
        if core_data['path'][0] =='o':
            continue
        else:
            
            filename = core_data['path'].split("/")[1].split(".")[0]
            all_file_name.append(filename)
            train_image_name = dataset_ids("train")
            test_image_name = dataset_ids("val")

            if filename in train_image_name:
                with open(f"../Data/anno_txt/train/{filename}.txt","w") as file:
                    for i in range(len(core_data['objects'])):
                        object_name = core_data['objects'][i]['category']
                        bBox = list(core_data['objects'][i]['bbox'].values())
                        file.write(f"{object_name} {round(bBox[0],7)} {round(bBox[1],7)} {round(bBox[2],7)} {round(bBox[3],7)} \n")

            elif filename in test_image_name:
                with open(f"../Data/anno_txt/val/{filename}.txt","w") as file:
                    for i in range(len(core_data['objects'])):
                        object_name = core_data['objects'][i]['category']
                        bBox = list(core_data['objects'][i]['bbox'].values())
                        file.write(f"{object_name} {round(bBox[0],7)} {round(bBox[1],7)} {round(bBox[2],7)} {round(bBox[3],7)} \n")
        
    return all_file_name

#create txt for image that not contain BBox
def empty_txt():
    empty_image=[]
    all_file = json2txt()
    train_image_name = dataset_ids("train")
    val_image_name = dataset_ids("val")
    for i in train_image_name:
        if i in all_file:
            continue
        else:
            empty_image.append(i)
            f = open(f"../Data/anno_txt/train/{i}.txt","w")
            f.close()

    for i in val_image_name:
        if i in all_file:
            continue
        else:
            empty_image.append(i)
            f = open(f"../Data/anno_txt/val/{i}.txt","w")
            f.close()

    return empty_image


# txt2xml
def txt2xml(set):
    txtList=[]
    with open(f"../Data/{set}_ids_TT.txt",'r') as f:
        line = f.readlines()
        for i in range(len(line)):
            txt = line[i].split(" ")
            if txt[-1].endswith("\n"):
                txt[-1] = txt[-1][:-1]
            txtList.append(txt)
    
    nouseLabel = target_class_name()
    
    for i in txtList:
        objectList=[]
        with open(f"../Data/anno_txt/{set}/{i[0]}.txt",'r') as f:
            line = f.readlines()
            for x in range(len(line)):
                object = line[x].split(" ")
                if object[-1].endswith("\n"):
                    object.pop(-1)
                if object[0] not in nouseLabel:
                    continue
                else:
                    objectList.append(object)

        filename = f"{i[0]}"

        root = Element('annotation')
        SubElement(root,'folder').text = "Images_TT"
        SubElement(root,'filename').text = f"{filename}.jpg"
        SubElement(root,'path').text = f"/home/ai-healthcare/Desktop/YooSeok/TS2git/Data/Images_TT/{filename}.jpg"

        source = SubElement(root, 'source')
        SubElement(source,'database').text = "Unknown"

        size = SubElement(root, 'size')
        SubElement(size,'width').text = "2048"
        SubElement(size,'height').text = "2048"
        SubElement(size,'depth').text = "3"

        SubElement(root,'segmented').text = "0"

        for i in range(len(objectList)):
            object = SubElement(root, 'object')

            SubElement(object,'name').text = f"{objectList[i][0]}"
            SubElement(object,'pose').text = "Unspecified"
            SubElement(object,'truncated').text = "0"
            SubElement(object,'difficult').text = "0"

            bndbox = SubElement(object, 'bndbox')
            SubElement(bndbox,'xmin').text = f"{objectList[i][1]}"
            SubElement(bndbox,'ymin').text = f"{objectList[i][2]}"
            SubElement(bndbox,'xmax').text = f"{objectList[i][3]}"
            SubElement(bndbox,'ymax').text = f"{objectList[i][4]}"

        dump(root)

        tree = ElementTree(root)
        tree.write('../Data/anno_xml/'+filename + '.xml',encoding='utf-8', xml_declaration=True)



# def main():
#     empty_txt()
#     txt2xml("val")
#     txt2xml("train")

    



# # before start you should remove image(35542,62778,78585,79029,88586,90422) in test set because it was overlap with train set
# if __name__ == "__main__":
#     main()
