import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import os
class NaiveBayes:
    def __init__(self):
        """Initializes the NaiveBayes classifier."""
        self.n_classes = 0
        self.priors = None
        self.cond_params = None
        self.idx_mp = []
        self.word_freqs = [] # To store word frequencies for word clouds
        self.word_mp = {}

    def fit(self, df, smoothening=1.0, class_col="Class Index", text_col="Tokenized Description"):
        """Learn the parameters of the model from the training data."""
        self.n_classes = df[class_col].nunique()
        
        # Initialize parameters
        self.cond_params = [[] for _ in range(self.n_classes)]
        self.cond_params_denom = np.zeros(self.n_classes)
        self.priors = np.zeros(self.n_classes)
        self.word_freqs = [{} for _ in range(self.n_classes)]
        
        self.word_mp = {}
        self.idx_mp = []
        next_word_idx = 0

        # First pass: count words, classes and build vocabulary
        for _, row in df.iterrows():
            text = row[text_col]
            c = row[class_col] - 1

            self.cond_params_denom[c] += len(text)
            self.priors[c] += 1
            for word in text:
                if word not in self.word_mp:
                    self.idx_mp.append(word)
                    self.word_mp[word] = next_word_idx
                    next_word_idx += 1
                    for i in range(self.n_classes):
                        self.cond_params[i].append(0)
                
                word_idx = self.word_mp[word]
                self.cond_params[c][word_idx] += 1
        
        # Store word frequencies for word clouds
        for i in range(self.n_classes):
            for word, idx in self.word_mp.items():
                count = self.cond_params[i][idx]
                if count > 0:
                    self.word_freqs[i][word] = count

        # Second pass: calculate conditonal paramters with smoothening
        vocab_size = len(self.word_mp)
        for i in range(self.n_classes):
            for j in range(vocab_size):
                self.cond_params[i][j] = (self.cond_params[i][j] + smoothening) / (self.cond_params_denom[i] + smoothening * vocab_size)
        
        self.priors = self.priors / len(df)
        self.cond_params = np.array(self.cond_params)

    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """Predict the class of the input data."""
        predictions = []
        for _, row in df.iterrows():
            log_probs = np.log(self.priors)
            text = row[text_col]
            for word in text:
                if word in self.word_mp:
                    word_idx = self.word_mp[word]
                    log_probs += np.log(self.cond_params[:, word_idx])
            
            predictions.append(np.argmax(log_probs) + 1)
        
        df[predicted_col] = predictions
        return df
    
    
def preprocess_content(df):
    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1
    processed_df['Tokenized Description'] = processed_df['content'].apply(
        lambda x: [word for word in x.lower().split()]
    )
    return processed_df

def preprocess_content_unigrams(df, stop_words,ps):

    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1

    def tokenize_unigrams(text):
        tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
        return tokens

    processed_df['Tokenized Description'] = processed_df['content'].apply(tokenize_unigrams)
    return processed_df

def preprocess_content_unigrams_bigrams(df,stop_words,ps):

    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1

    def tokenize_unigrams_bigrams(text):
        tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
        bigrams = ['_'.join(bi) for bi in ngrams(tokens, 2)]
        return tokens + bigrams

    processed_df['Tokenized Description'] = processed_df['content'].apply(tokenize_unigrams_bigrams)
    return processed_df

def preprocess_title(df):
    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1
    processed_df['Tokenized Description'] = processed_df['title'].apply(
        lambda x: [word for word in x.lower().split()]
    )
    return processed_df

def preprocess_title_unigrams(df, stop_words,ps):

    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1

    def tokenize_unigrams(text):
        tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
        return tokens

    processed_df['Tokenized Description'] = processed_df['title'].apply(tokenize_unigrams)
    return processed_df

def preprocess_title_unigrams_bigrams(df,stop_words,ps):

    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1

    def tokenize_unigrams_bigrams(text):
        tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
        bigrams = ['_'.join(bi) for bi in ngrams(tokens, 2)]
        return tokens + bigrams

    processed_df['Tokenized Description'] = processed_df['title'].apply(tokenize_unigrams_bigrams)
    return processed_df

def preprocess_title_content_concat(df, stop_words,ps):
    processed_df = df.copy()
    processed_df['Class Index'] = processed_df['label'] + 1

    def tokenize_unigrams_bigrams(text):
        tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
        bigrams = ['_'.join(bi) for bi in ngrams(tokens, 2)]
        return tokens + bigrams

    processed_df['Tokenized Description'] = processed_df['title'].apply(tokenize_unigrams_bigrams)+processed_df['content'].apply(tokenize_unigrams_bigrams)
    return processed_df

def tokenize_unigrams_bigrams(text, stop_words, ps):
    tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
    bigrams = ['_'.join(bi) for bi in ngrams(tokens, 2)]
    return tokens + bigrams

def tokenize_n_grams(text,stop_words,ps):
    tokens = [ps.stem(word) for word in text.lower().split() if word not in stop_words]
    bigrams = ['_'.join(bi) for bi in ngrams(tokens, 2)]
    trigrams=['_'.join(tri) for tri in ngrams(tokens, 3)]
    quadgrams=['_'.join(quad) for quad in ngrams(tokens, 4)]
    pentgrams=['_'.join(pent) for pent in ngrams(tokens, 5)]
    return tokens + bigrams +trigrams+quadgrams+pentgrams

def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Generates a random color for the word cloud."""
    return f"hsl({random.randint(0, 255)}, {random.randint(60, 100)}%, {random.randint(40, 60)}%)"

def make_wordcloud(frequencies, save_name, output_dir):
    """
     Saves wordcloud images in the output_dir
    """
    if not frequencies:
        print(f"Skipping '{save_name}' because there are no words for this class.")
        return

    # make output dir incase it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # create wordcloud using the lib
    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=400,
        max_words=150,
        scale=2,
        random_state=42
    ).generate_from_frequencies(frequencies)
    # saving in path
    wordcloud.recolor(color_func=random_color_func)
    save_path = os.path.join(output_dir, save_name)
    wordcloud.to_file(save_path)
    print(f"Generated word cloud: {save_path}")

def predict_dual(df, nb_title, nb_content):
    predictions = []
    for _, row in df.iterrows():
        title = row["Tokenized Title"]
        content = row["Tokenized Content"]

        # initalize with 
        log_probs = np.log(nb_title.priors)

        # Add title cond_params
        for word in title:
            if word in nb_title.word_mp:
                idx = nb_title.word_mp[word]
                log_probs += np.log(nb_title.cond_params[:, idx])

        # Add content cond_params
        for word in content:
            if word in nb_content.word_mp:
                idx = nb_content.word_mp[word]
                log_probs += np.log(nb_content.cond_params[:, idx])

        # make the prediction as the class with highest log_probs
        predictions.append(np.argmax(log_probs) + 1)

    df["Predicted"] = predictions
    return df

def train_evaluate_model(train_df,test_df,output_dir):

    nb_model = NaiveBayes()
    nb_model.fit(train_df)

    # Generate Word Clouds 
    print("Generating Word Clouds for Each Class")
    # Get the unique cls labels from the df
    class_labels = sorted(train_df['Class Index'].unique())
    for i, class_label in enumerate(class_labels):
        frequencies = nb_model.word_freqs[i]
        filename = f"wordcloud_class_{class_label}.png"
        make_wordcloud(frequencies, filename,output_dir)

    # predict on train data
    train_predictions = nb_model.predict(train_df.copy())
    
    # predict on test data
    test_predictions = nb_model.predict(test_df.copy())
    # get accuracy
    train_correct = (train_predictions['Predicted'] == train_predictions['Class Index']).sum()
    train_accuracy = train_correct / len(train_predictions)

    test_correct = (test_predictions['Predicted'] == test_predictions['Class Index']).sum()
    test_accuracy = test_correct / len(test_predictions)
    # Print evaluation metrics
    print("Training Set Metrics")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(classification_report(train_df['Class Index'], train_predictions['Predicted'], digits=3))

    print("Test Set Metrics")
    print(f"Testing Accuracy: {test_accuracy:.2%}")
    print(classification_report(test_df['Class Index'], test_predictions['Predicted'], digits=3))

if __name__ == "__main__":
    """
    Training model on raw content data
    """
    print("Raw content data model")
    # Load and Prepare Data
    train_df = pd.read_csv(r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\train.csv")
    test_df=pd.read_csv(r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\test.csv")

    # Preprocess Data 
    train_raw = preprocess_content(train_df)
    test_raw = preprocess_content(test_df)

    # Train and evaluate the Model
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\raw_content_data_wordcloud"
    train_evaluate_model(train_raw,test_raw,output_dir)
    # Define libraries for stop words and stemming
    stop_words = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', "couldn't", 'did', "didn't", 'do',
    'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
    'he', "he'd", "he'll", "he's", 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is',
    "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'just', 'll', 'm', 'ma',
    'me', "mightn't", 'more', 'most', "mustn't", 'my', 'myself', "needn't", 'no',
    'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', "shan't", 'she',
    "she'd", "she'll", "she's", 'should', "shouldn't", "should've", 'so', 'some',
    'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're",
    "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've',
    'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were',
    "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', "won't", "wouldn't", 'y', 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves'
}
    ps = PorterStemmer()
    
    """
    Training model on processed content data
    """

    print("Processed content data model")
    # Preprocess Data 
    train_processed = preprocess_content_unigrams(train_df,stop_words,ps)
    test_processed = preprocess_content_unigrams(test_df,stop_words,ps)

    # Train the Model and evaluate
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\processed_content_data_wordcloud"
    train_evaluate_model(train_processed,test_processed,output_dir)
    
    """
    Training on bigrams content model
    """
    print("Training on bigram+unigram content model")
    train_bigram_processed = preprocess_content_unigrams_bigrams(train_df,stop_words,ps)
    test_bigram_processed = preprocess_content_unigrams_bigrams(test_df,stop_words,ps)

    # Train and evaluate the Model
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\processed_bigram_content_data_wordcloud"
    train_evaluate_model(train_bigram_processed,test_bigram_processed,output_dir)

    """
    Training model on raw title data
    """
    print("Raw title data model")
    # Preprocess Data 
    train_raw = preprocess_title(train_df)
    test_raw = preprocess_title(test_df)

    # Train and evaluate the Model
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\raw_title_data_wordcloud"
    train_evaluate_model(train_raw,test_raw,output_dir)

    """
    Training model on processed title data
    """

    print("Processed title data model")
    # Preprocess Data 
    train_processed = preprocess_title_unigrams(train_df,stop_words,ps)
    test_processed = preprocess_title_unigrams(test_df,stop_words,ps)

    # Train the Model and evaluate
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q1\processed_title_data_wordcloud"
    train_evaluate_model(train_processed,test_processed,output_dir)

    """
    Training on bigrams title model
    """
    print("Training on bigram+unigram title model")

    # Preprocess data
    train_bigram_processed = preprocess_title_unigrams_bigrams(train_df,stop_words,ps)
    test_bigram_processed = preprocess_title_unigrams_bigrams(test_df,stop_words,ps)

    # Train and evaluate the Model
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q2\processed_bigram_title_data_wordcloud"
    train_evaluate_model(train_bigram_processed,test_bigram_processed,output_dir)

    """
    Training on title and content concat model
    """
    print("Training on title and content concat model")
    # Preprocess data
    train_concat=preprocess_title_content_concat(train_df,stop_words,ps)
    test_concat=preprocess_title_content_concat(test_df,stop_words,ps)
    # Train and evaluate model
    output_dir=r"C:\Users\jayes\Desktop\CP\Assignment 2\Q2\processed_concat_wordcloud"
    train_evaluate_model(train_concat,test_concat,output_dir)

    """
    MLE using two models trained on title and content seperately
    """
    # Preprocess Data 
    print("MLE using two models")
    train_processed = train_df.copy()
    train_processed['Class Index'] = train_processed['label'] + 1
    train_processed['Tokenized Title'] = train_processed['title'].apply(lambda t: tokenize_unigrams_bigrams(t, stop_words, ps))
    train_processed['Tokenized Content'] = train_processed['content'].apply(lambda c: tokenize_unigrams_bigrams(c, stop_words, ps))

    test_processed = test_df.copy()
    test_processed['Class Index'] = test_processed['label'] + 1
    test_processed['Tokenized Title'] = test_processed['title'].apply(lambda t: tokenize_unigrams_bigrams(t, stop_words, ps))
    test_processed['Tokenized Content'] = test_processed['content'].apply(lambda c: tokenize_unigrams_bigrams(c, stop_words, ps))

    # Train two separate NaiveBayes models (title & content)
    nb_title = NaiveBayes()
    nb_title.fit(train_processed, text_col="Tokenized Title", class_col="Class Index")

    nb_content = NaiveBayes()
    nb_content.fit(train_processed, text_col="Tokenized Content", class_col="Class Index")

    # Predict 
    train_predictions = predict_dual(train_processed.copy(), nb_title, nb_content)
    test_predictions  = predict_dual(test_processed.copy(),  nb_title, nb_content)

    #  Compute evaluation metrics
    train_acc = (train_predictions['Predicted'] == train_predictions['Class Index']).mean()
    test_acc  = (test_predictions['Predicted']  == test_predictions['Class Index']).mean()

    print("Training Set Metrics")
    print(f"Training Accuracy: {train_acc:.2%}")
    print(classification_report(train_processed['Class Index'], train_predictions['Predicted'], digits=3))

    print("Test Set Metrics")
    print(f"Testing Accuracy: {test_acc:.2%}")
    print(classification_report(test_processed['Class Index'], test_predictions['Predicted'], digits=3))
    cm = confusion_matrix(test_processed['Class Index'], test_predictions['Predicted'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.show()
    # Preprocess Data s
    print("Feature engineering on best model")
    train_processed = train_df.copy()
    train_processed['Class Index'] = train_processed['label'] + 1
    train_processed['Tokenized Title'] = train_processed['title'].apply(lambda t: tokenize_n_grams(t, stop_words, ps))
    train_processed['Tokenized Content'] = train_processed['content'].apply(lambda c: tokenize_n_grams(c, stop_words, ps))

    test_processed = test_df.copy()
    test_processed['Class Index'] = test_processed['label'] + 1
    test_processed['Tokenized Title'] = test_processed['title'].apply(lambda t: tokenize_n_grams(t, stop_words, ps))
    test_processed['Tokenized Content'] = test_processed['content'].apply(lambda c: tokenize_n_grams(c, stop_words, ps))

    # train two seperate models, one on title and another one on content
    nb_title = NaiveBayes()
    nb_title.fit(train_processed, text_col="Tokenized Title", class_col="Class Index")

    nb_content = NaiveBayes()
    nb_content.fit(train_processed, text_col="Tokenized Content", class_col="Class Index")

    # predict using both models
    train_predictions = predict_dual(train_processed.copy(), nb_title, nb_content)
    test_predictions  = predict_dual(test_processed.copy(),  nb_title, nb_content)

    #  compute the acc
    train_acc = (train_predictions['Predicted'] == train_predictions['Class Index']).mean()
    test_acc  = (test_predictions['Predicted']  == test_predictions['Class Index']).mean()

    print("training Set Metrics")
    print(f"Training Accuracy: {train_acc:.2%}")
    print(classification_report(train_processed['Class Index'], train_predictions['Predicted'], digits=3))

    print("test Set Metrics ")
    print(f"Testing Accuracy: {test_acc:.2%}")
    print(classification_report(test_processed['Class Index'], test_predictions['Predicted'], digits=3))






