import os
import gc
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

get_ipython().run_line_magic('matplotlib', 'inline')


def plot(plotpath):
    directory_list = os.listdir(plotpath)
    for x in directory_list:
        for _, row in train[train.filepath == plotpath + x].iterrows():
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            image = plt.imread(plotpath + x)
            plt.axis('off')
            plt.imshow(image)
            xmin = row.xmin
            xmax = row.xmax
            ymin = row.ymin
            ymax = row.ymax

            width = xmax - xmin
            height = ymax - ymin

            # assign different color to different classes of objects
            edgecolor = 'g'
            ax.annotate('Phone', xy=(xmax, ymin))
            # add bounding boxes to the image
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')
            ax.add_patch(rect)
            y = x.split(".jpg")
            plt.savefig("./plottedimages/" + y[0] + ".png", bbox_inches="tight", pad_inches=0)
            # Clear the current axes.
            plt.cla()
            # Clear the current figure.
            plt.clf()
            plt.close(fig)

    print("done")


train = pd.read_csv("train_anno_kaggle.csv")
train['filepath'].nunique()
train['class_name'].value_counts()
datasetpath = "C:/Users/Zaid/Desktop/Datasets/Kaggle/"
plotpath1 = datasetpath + "phonetrain/c1/"
plotpath2 = datasetpath + "phonetrain/c2/"
plotpath3 = datasetpath + "phonetrain/c3/"
plotpath4 = datasetpath + "phonetrain/c4/"
# doing the plotting
plot(plotpath1)
gc.collect()
plot(plotpath2)
gc.collect()
plot(plotpath3)
gc.collect()
plot(plotpath4)
gc.collect()
