from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import sys


def loadData(filename):
    sentences = []
    with open(filename, encoding='iso8859-15') as f:
        sent = []
        for line in f:
            line = line.strip()
            if (len(line) == 0 ):
                if len(sent) != 0:
                    sentences.append(sent)
                    sent = []
            else:
                #ls = line.split(' ')
                #word, tag = ls[0],ls[-1]
                ls = line.strip()
                word = ls
                sent.append(word)
    return sentences

def writeData(filename,sentences):
    with open(filename,"w") as f:
        for sent in sentences:
            for tuple in sent:
                f.write(tuple[0] + " "+ tuple[1]+"\n")
            f.write("\n")

def evaluate(model,sentences):
    labelledSentences = []
    for sent in sentences:
        preds = model.predict(sent)
        single_label_sent = []
        for (word,tag) in zip(sent,preds):
            single_label_sent.append(((word,tag)))
        labelledSentences.append(single_label_sent)
    return labelledSentences


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    print("Model Restored")

    config.filename_test = sys.argv[1]
    filename_test_label = config.filename_test+".labelled";

    print("Loading Test File")
    sentences = loadData(config.filename_test)
    print("Evaluating Model on Test File")
    labelledSentences = evaluate(model,sentences)
    print("Saving Word,Tag in Labelled Files")
    writeData(filename_test_label,labelledSentences)


if __name__ == "__main__":
    main()
