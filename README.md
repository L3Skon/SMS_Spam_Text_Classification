# SMS_Spam_Text_Classification

Dataset src : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Text Preprocessing, Visualization, and Classification: A Comprehensive Approach for Spam Message Analysis

The required libraries are imported, including pandas for data manipulation, nltk for natural language processing, and various modules from scikit-learn and Keras for machine learning and deep learning.

The NLTK stopwords corpus is downloaded to remove common English stopwords from the text.

A CSV file named 'spam.csv' is read into a pandas DataFrame. Only the 'v1' (category) and 'v2' (message) columns are selected and renamed.

Data preprocessing steps are performed on the DataFrame: duplicate rows are dropped, rows with missing values are removed, and the 'Message' column is converted to lowercase and cleaned by removing non-alphanumeric characters.

Tokenization is applied to the 'Message' column, splitting each message into a list of words.

Stopwords are removed from the tokenized messages.

Stemming is performed on the tokenized messages using the Porter stemming algorithm, reducing words to their base or root form.

The tokenized and processed messages are converted back to text by joining the words with spaces.

Frequency distribution of the unigrams (individual words) in the text is calculated and the top 10 most common unigrams are printed.

Data visualization is performed using the Seaborn library to display the distribution of spam and non-spam messages and the frequency of the top 20 words.

The dataset is split into training and testing sets using a 80:20 ratio.

The training set is transformed using TF-IDF vectorization, which assigns weights to words based on their importance in the documents.

The TF-IDF vectors are normalized using the L2 norm.

The target labels are encoded using label encoding.

A K-Nearest Neighbors classifier is trained on the TF-IDF vectors and used to predict the labels of the test set. The performance metrics (AUC, precision, recall, and F1-score) are calculated and printed.

A confusion matrix is plotted using the predicted labels and true labels of the test set.

Tokenization is applied to the text messages in the training set using the Keras Tokenizer.

The vocabulary size and maximum sequence length are determined.

The text messages are converted to sequences and padded to have the same length.

A deep learning model with an embedding layer, LSTM layer, and a dense output layer is defined and compiled.

The model is trained on the padded sequences and evaluated on the test set. The AUC score is calculated and printed.

The predictions from the LSTM model are converted to strings and a DataFrame is created with the true labels and predicted labels.

A confusion matrix is plotted using the predicted labels and true labels of the test set for the LSTM model.

