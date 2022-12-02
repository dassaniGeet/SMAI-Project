from adaboost import *

class Cascade:
    def __init__(self, layer):
        self.clfs = []
        self.layers = layer

    def classify(self, image):
        flag=0
        for clf in self.clfs:
            if clf.classify(image) != 0:
                flag=0
            else:
                flag=1
                break
        if(flag==1):
            return 0
        else:
            return 1

    def train(self, training):
        positive = []
        negative=[]

        for i in training:
            if (i[1]!=1):
                negative.append(i)
            else:
                positive.append(i)
        
        for fn in self.layers:
            if(len(negative)!=0):
                clf = vclassifier(fn)
                clf.train(positive+negative, len(positive), len(negative))
                self.clfs.append(clf)
                false_positives = []
                for ex in negative:
                    if self.classify(ex[0]) == 1:
                        false_positives.append(ex)
                negative = false_positives
            else:
                print("Stopped Before. False Positive Ratio(FPR) = 0")
                break