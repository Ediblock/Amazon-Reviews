import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt

def cleanText(data_text:list, regEx_acceptable_characters:str = None):
    """Cleans and prepare data text for the network.
    It accepts list of strings"""
    if regEx_acceptable_characters is None:
        for itr, line in enumerate(data_text):
            data_text[itr]= re.sub("https.\S+", "", line.lower())       #deleting websites
    else:
        for itr, line in enumerate(data_text):
            data_text[itr] = re.sub("https.\S+", "", line)  # deleting websites
            data_text[itr] = re.sub(f"[^{regEx_acceptable_characters}]", "", line.lower())
    return data_text

def formatText(text:list):
    """Formats text to data and labels. labels '__label__1' and '__label__2' will be converted to more readable format
     as 0 and 1. 0 would be bad review and 1 will be positive review for sentiment analysis

    It accepts format list of strings"""
    labels = []
    data = []
    for textLine in text:
        labelAndDataSpan = re.search("\s", textLine).span()
        if "__label__1" in textLine[0:labelAndDataSpan[0]]:
            labels.append(0)
        elif "__label__2" in textLine[0:labelAndDataSpan[0]]:
            labels.append(1)
        data.append(textLine[labelAndDataSpan[1]:][::-1])
    return data, labels

def dataLengthHistogram(data_text:list):
    """Shows on the histogram how much words is in the reviews.
    Accepts list of texts"""
    line_text_length_list = []
    #Counting characters in each text line
    for line_text in data_text:
        line_text_length_list.append(len(line_text))

    fig = plt.figure()
    ax_hist = fig.add_subplot(111)
    ax_hist.hist(line_text_length_list, 50)
    ax_hist.set_xlabel("Number of characters in sentence")
    ax_hist.set_ylabel("Number of sentences per characters")
    plt.show()
    print("Mean is: %d" % np.mean(line_text_length_list))
    print("Median is: %s" % np.median(line_text_length_list))

def findUniqueCharacters(data_text:list):
    unique_characters_list = []
    for line_text in data_text:
        unique_characters_list += list(set(line_text))

    unique_characters_list = list(set(unique_characters_list))
    print(unique_characters_list)
    print(len(unique_characters_list))
    unique_characters_list.sort()
    print(unique_characters_list)
    print("".join(unique_characters_list))
    with open("./data/uniqueList.txt", "w", encoding="utf-8") as fd:
        fd.write("".join(unique_characters_list))

def createTensorForNetworkFromText(data_text:list, list_of_allowed_characters:str, character_length:int) -> np.ndarray:
    """This function creates tensor of all the texts with one hot encoding from list_of_allowed_characters.
    Data for line of text in tensor will look like (1, len(list_of_allowed_characters), character_length, 1).
    One hot encoding on characters is made form list of characters in "list_of_allowed_characters".

    Texts longer then "character_length" will be truncated.
    Texts shorter then "character_length" will be filled with zeros.

    Returns numpy array as 3 dimensional"""
    dataArray = np.zeros((len(data_text), len(list_of_allowed_characters), character_length), dtype=np.int8)
    for i, text in enumerate(data_text):
        for j, character in enumerate(list_of_allowed_characters):
            if len(text) > character_length:
                for k in range(character_length):
                    if character in text[k]:
                        dataArray[i, j, k] = 1
            else:
                for k, text_character in enumerate(text):
                    if character in text_character:
                        dataArray[i, j, k] = 1
    return dataArray

def createModel(inputShape3D:tuple) -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Convolution1D(inputShape3D[2], kernel_size=4, input_shape=inputShape3D[1:], activation="relu"))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.Convolution1D(512, kernel_size=4, activation="relu"))
    model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Convolution1D(256, kernel_size=4, activation="relu"))
    model.add(tf.keras.layers.Convolution1D(128, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.Convolution1D(64, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.Convolution1D(128, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.Convolution1D(256, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation="relu"))
    model.add(tf.keras.layers.Dense(1024, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

class ManageHugeDatasetsWithGenerator(tf.keras.utils.Sequence):
    def __init__(self, filePath:str, strOfAcceptableCharacters:str, numberOfCharactersInText:int, batchSize:int = 32,
                 shuffle = True):
        self.filePath = filePath
        self.batchSize = batchSize
        self.strOfAcceptableCharacters = strOfAcceptableCharacters
        self.numberOfCharactersInText = numberOfCharactersInText
        self.shuffle = shuffle
        with open(self.filePath, "r", encoding="utf-8") as text:
            self.dataTextRaw = text.readlines()
        self.on_epoch_end()


    def __cleanText(self, data_text:list, regEx_acceptable_characters:str):
        """Cleans and prepare data text for the network.
        It accepts list of strings"""
        if regEx_acceptable_characters is None:
            for itr, line in enumerate(data_text):
                data_text[itr] = re.sub("https.\S+", "", line.lower())  # deleting websites
        else:
            for itr, line in enumerate(data_text):
                data_text[itr] = re.sub("https.\S+", "", line)  # deleting websites
                data_text[itr] = re.sub(f"[^{regEx_acceptable_characters}]", "", line.lower())
        return data_text

    def __formatText(self, text: list):
        """Formats text to data and labels. labels '__label__1' and '__label__2' will be converted to more readable format
         as 0 and 1. 0 would be bad review and 1 will be positive review for sentiment analysis

        It accepts format list of strings"""
        labels = []
        data = []
        for textLine in text:
            labelAndDataSpan = re.search("\s", textLine).span()
            if "__label__1" in textLine[0:labelAndDataSpan[0]]:
                labels.append(0)
            elif "__label__2" in textLine[0:labelAndDataSpan[0]]:
                labels.append(1)
            data.append(textLine[labelAndDataSpan[1]:][::-1])
        return data, labels

    def __len__(self):
        "Denotes number of batches per epoch"
        return int(np.floor(len(self.dataTextRaw)/self.batchSize))

    def on_epoch_end(self):
        "Update indexes after each epoch"
        self.indexes = np.arange(len(self.dataTextRaw))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, listOfIndexes:list):
        """Generate data containing batch size sample # (n samples, len(self.strOfAcceptableCharacters), self.numberOfCharactersInText)
        This function creates tensor of all the texts with one hot encoding from list_of_allowed_characters.
        Data for line of text in tensor will look like (1, len(list_of_allowed_characters), character_length, 1).
        One hot encoding on characters is made form list of characters in "list_of_allowed_characters".

        Texts longer then "character_length" will be truncated.
        Texts shorter then "character_length" will be filled with zeros"""
        dataArrayBatch = np.zeros((self.batchSize, len(self.strOfAcceptableCharacters), self.numberOfCharactersInText), dtype="int8")
        dataTextSlice = []
        for i in listOfIndexes:
            dataTextSlice.append(self.dataTextRaw[i])
        dataTextSlice, dataLabelsSlice = self.__formatText(self.__cleanText(dataTextSlice, self.strOfAcceptableCharacters))
        for i, text in enumerate(dataTextSlice):
            for j, character in enumerate(self.strOfAcceptableCharacters):
                if len(text) > self.numberOfCharactersInText:
                    for k in range(self.numberOfCharactersInText):
                        if text[k] == character:
                            dataArrayBatch[i, j, k] = 1
                else:
                    for k, textCharacter in enumerate(text):
                        if textCharacter == character:
                            dataArrayBatch[i, j, k] = 1

        return dataArrayBatch, np.array(dataLabelsSlice, dtype="int8")

    def __getitem__(self, index):
        "Generate one batch of data"
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batchSize:(index + 1)*self.batchSize]

        X, y = self.__data_generation(indexes)
        return X, y




    # def batch_generator(self, slice_number:int, string_of_allowable_characters:str, number_of_characters_in_text:int):
    #     with open("./data/train.ft.txt", "r", encoding="utf-8") as text:
    #         file_text_raw = text.readlines()
    #
    #     #getting right slice number if the module is not equal to 0
    #     while True:
    #         if len(file_text_raw)%slice_number == 0:
    #             break
    #         else:
    #             slice_number = slice_number + 1
    #     numberOfDataInSlice = np.floor(len(file_text_raw)/slice_number)
    #     dataArray = np.zeros((numberOfDataInSlice, len(string_of_allowable_characters), number_of_characters_in_text), dtype="int8")

with open("./data/uniqueList.txt", "r") as fd:
    list_acceptable_characters = fd.read()
# train_categorical_labels = tf.keras.utils.to_categorical(train_labels, 1)

NUMBER_CHARACTERS_IN_TEXT = 1024
batchSize = 360
try:
    model = tf.keras.models.load_model("./model")
except:
    # Created model not trained yet
    model = createModel((None, len(list_acceptable_characters), NUMBER_CHARACTERS_IN_TEXT))
    #Training model
    filePath = "./data/train.ft.txt"
    # with open("./data/train.ft.txt", "r", encoding="utf-8") as text:
    #     file_text_raw = text.readlines()
    # train_data_raw, train_labels = formatText(cleanText(file_text_raw, list_acceptable_characters))
    # train_data_raw = createTensorForNetworkFromText(train_data_raw, list_acceptable_characters, NUMBER_CHARACTERS_IN_TEXT)
    # train_data_raw = tf.data.Dataset.from_tensor_slices((train_data_raw, np.array(train_labels, dtype="int8")))

    modelFilePath = "./checkpointModel"
    saveModelCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=modelFilePath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        save_freq="epoch"
    )

    train_data_generator = ManageHugeDatasetsWithGenerator(filePath=filePath, strOfAcceptableCharacters=list_acceptable_characters,
                                              numberOfCharactersInText=NUMBER_CHARACTERS_IN_TEXT, batchSize=batchSize)
    model.fit(train_data_generator, epochs=20, verbose=1, callbacks=[saveModelCallback])
    # for i in range(numberOfTrainingIterations):
    #     training_data_cut = createTensorForNetworkFromText(train_data_raw[int((i * len(train_data_raw)) / numberOfTrainingIterations):
    #                                                                       int(((i + 1) * len(train_data_raw)) / numberOfTrainingIterations - 1)],
    #                                                        list_acceptable_characters, NUMBER_CHARACTERS_IN_TEXT)
    #     labels_cut = train_labels[int((i*len(train_labels))/numberOfTrainingIterations):
    #                               int(((i + 1)*len(train_labels)/numberOfTrainingIterations) - 1)]
    #     # print("Iteration number: %d\nLength of training %d\nLength of labels %d" % (i, len(training_data_cut), len(labels_cut)))
    #     # print("Start position of training batch: %d, Stop position of training batch: %d" % (int((i*len(train_data_raw))/numberOfTrainingIterations),
    #     #                                                                                      int(((i + 1)*len(train_data_raw))/numberOfTrainingIterations - 1)))
    #     labels_cut = np.array(labels_cut, dtype="int8")
    #     model.fit(training_data_cut, labels_cut, batch_size=32, epochs=20, verbose=1)
    # tf.keras.models.save_model(model, "./model")

#Evaluating model
with open("./data/test.ft.txt", "r", encoding="utf-8") as test_text:
    file_test_text_raw  = test_text.readlines()

numberOfTrainingIterations = 20

test_data_raw, test_labels = formatText(cleanText(file_test_text_raw, list_acceptable_characters))

test_data_generator = ManageHugeDatasetsWithGenerator("./data/test.ft.txt", list_acceptable_characters, NUMBER_CHARACTERS_IN_TEXT, batchSize=400)
metrics = model.evaluate(test_data_generator, verbose=1)
print()
print("%s : %.2f%%" % (model.metrics_names[1], metrics[1] * 100))





