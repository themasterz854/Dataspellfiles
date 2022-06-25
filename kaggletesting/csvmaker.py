import csv
import os

header = ['filepath', 'xmin', 'ymin', 'xmax', 'ymax', 'class_name']
data = []
datasetpath = "C:/Users/Zaid/Desktop/Datasets/Kaggle/"
imgpath1 = datasetpath + "nophonetrain/c0/"
imgpath2 = datasetpath + "nophonetrain/c5/"
imgpath3 = datasetpath + "nophonetrain/c6/"
imgpath4 = datasetpath + "nophonetrain/c7/"
imgpath5 = datasetpath + "nophonetrain/c8/"
imgpath6 = datasetpath + "nophonetrain/c9/"

imgpath = [imgpath1, imgpath2, imgpath3, imgpath4, imgpath5, imgpath6]
with open('combined.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    for lpath in imgpath:
        dl = os.listdir(lpath)
        for x in dl:
            data.append(os.path.abspath(lpath + x).replace("\\", "/"))
            data.append('0')
            data.append('0')
            data.append('0')
            data.append('0')
            data.append('no_phone')
            writer.writerow(data)
            data = []
    f.close()

with open('train_anno_kaggle.csv', 'r') as f:
    reader1 = csv.reader(f)
    with open('combined.csv', 'a', newline='') as f_object:
        writer_object = csv.writer(f_object)
        for row in reader1:
            writer_object.writerow(row)
        f_object.close()
    f.close()
    print("done")
