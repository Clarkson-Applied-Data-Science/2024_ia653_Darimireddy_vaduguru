# 2024_ia653_Darimireddy_vaduguru

## Project Overview and Process Narrative

### Project Overview
The primary objective of this project was to develop and compare different machine learning models for text classification based on sentiment analysis. The dataset used comprises user reviews, with the task being to classify these reviews based on sentiment (positive, neutral, negative). To enhance the robustness of the model and address potential data imbalances, synthetic "neutral" reviews were generated and integrated with the existing data.

### Process Narrative
The project began with setting up the necessary libraries and tools required for data manipulation, text processing, and model training. This setup included libraries like Pandas for data handling, NLTK for natural language processing, TensorFlow for building neural networks, and Scikit-learn for machine learning models.

#### Data Preprocessing
The initial phase involved loading the existing dataset and inspecting it for quality and structure. It became evident that the dataset lacked sufficient "neutral" reviews. To address this, I utilized a predefined set of sentence patterns and word pools to generate 15,000 synthetic neutral reviews, aiming to balance the dataset and improve model training outcomes.

### Dataset Description

#### Data Acquisition and Composition
The dataset for this project consisted of user reviews from Spotify, originally containing 52702 rows of data with various features such as review text and sentiment labels. To effectively train models on sentiment analysis and ensure robustness, it was crucial to have a well-balanced dataset regarding the distribution of sentiments (positive, neutral, negative).

#### Synthetic Data Generation
Given the initial dataset was skewed with an underrepresentation of neutral sentiments, synthetic data generation was employed to balance the dataset. Using a combination of predefined sentence patterns and a word pool, 15,000 neutral reviews were programmatically created. This method ensured the neutrality of the reviews by carefully selecting words that are typically considered neutral in sentiment.
1. Predefined Sentence Patterns:
The patterns list contains sentence templates with placeholders (e.g., {frequency}, {adjective}) that will be replaced by words or phrases from the word pools. These patterns resemble neutral statements about Spotify, ensuring the generated reviews are not overly positive or negative.

2. Word Pools:
The word_pools dictionary categorizes words or phrases to replace placeholders in the review patterns, ensuring the generated content remains neutral and generic. For instance, the "frequency" pool includes terms like "occasionally" and "frequently," describing how often Spotify is used. The "adjective" pool provides neutral descriptors such as "decent" or "fine" to characterize Spotify's features or performance. Similarly, the "fulfillment" pool contains phrases like "does its job" or "serves its purpose," emphasizing adequacy without being overly positive or critical. These pools help maintain a consistent tone across reviews, avoiding strong opinions while mimicking balanced user feedback.

3. Review Generation Function:
The generate_reviews function generates the specified number of reviews by first randomly selecting a pattern from the patterns list. Each placeholder in the chosen pattern is then replaced with a random value from the corresponding category in the word_pools dictionary. For example, a placeholder for "frequency" will be replaced with a word like "occasionally" or "frequently," while a placeholder for "adjective" will be substituted with terms like "decent" or "fine." After all placeholders are replaced, the function constructs a review as a dictionary, where the "Review" key contains the generated sentence, and the "label" key is set to "NEUTRAL." This process ensures that a variety of neutral reviews are created.

4. Repeat Process for All Reviews:
The loop runs n times to create the specified number of reviews. In your case, generate_reviews(15000) generates 15,000 neutral reviews.
The function returns a list of dictionaries, where each dictionary contains:
A "Review": The generated neutral sentence.
A "label": "NEUTRAL" indicating the sentiment of the review.

#### Data Structure and Enhancements
The augmented dataset includes features like the review text and associated sentiment labels. To integrate the synthetic reviews seamlessly with the original data, they were concatenated, and the entire dataset was shuffled to ensure random distribution before splitting into training and testing sets.

#### Preprocessing Details
1. *Cleaning:* The review texts were cleaned to remove any irrelevant characters, standardize the text format, and prepare them for vectorization.
2. *Tokenization:* Text data was tokenized to convert sentences into words, allowing for further processing such as vectorization.
3. *Vectorization:* Both CountVectorizer and TF-IDF Vectorizer were used in different model setups to convert text data into a numerical format that machine learning algorithms can process.
4. *Label Encoding:* Sentiment labels were encoded into numerical values to facilitate model training and evaluation.

#### Data Integrity and Balancing
To address potential issues of data imbalance which could bias the model performance towards the majority class, the dataset was balanced by the addition of synthetic reviews. Moreover, care was taken to ensure that no data leakage occurred between the training and testing sets, maintaining the integrity of the model evaluation process.

### Exploratory Data Analysis (EDA) and Preprocessing Steps

#### EDA Overview
Exploratory Data Analysis was conducted to gain insights into the dataset’s characteristics and to guide the subsequent preprocessing steps. The EDA involved examining the distribution of sentiment labels, identifying any anomalies or outliers in the data, and understanding the textual content through various linguistic analyses.

#### Key Findings from EDA
1. *Label Distribution:* Initial analysis showed a significant imbalance with an underrepresentation of neutral sentiments, which led to the decision to synthesize additional neutral reviews.
2. *Text Content Analysis:* Analysis of the text content included identifying common words and phrases. Utilizing NLTK, the most frequent words were examined, and stopwords were identified for removal to reduce noise in the data.

#### Preprocessing Steps
Based on the insights gained from the EDA, several preprocessing steps were implemented to prepare the data for modeling:

1. *Text Cleaning:* This included removing special characters, numbers, and punctuation from the reviews to standardize the text data.
2. *Tokenization:* Texts were broken down into individual words or tokens, facilitating easier manipulation and analysis.
3. *Stopword Removal:* Commonly used words that do not contribute significant meaning to the sentences (like "the", "is", "at") were removed.
4. *Lemmatization:* Words were reduced to their base or root form, smoothing out the complexity and improving the model’s ability to understand the essence of the text.
5. *Vectorization:* To convert text data into a numerical format that could be fed into machine learning models, vectorization techniques such as Count Vectorization and TF-IDF Vectorization were applied. This step is crucial as it transforms raw text into a structured format suitable for modeling.

#### Handling Imbalanced Data
To ensure the models did not develop a bias toward the majority class:
- *Synthetic Review Integration:* As mentioned, synthetic neutral reviews were generated and added to the dataset to balance the distribution of sentiments.
- *Stratified Sampling:* During the train-test split, stratified sampling was employed to maintain an even distribution of sentiment labels across both training and testing datasets.

These preprocessing efforts laid the groundwork for effective model training, ensuring that the input data was clean, structured, and well-suited for the machine learning tasks ahead. The thorough preprocessing also aimed to enhance model performance by providing clear and relevant features from the text data.

### Model Fitting and Validation

#### Train/Test Splitting
The dataset was divided into training and testing sets using a stratified sampling approach to maintain an equal proportion of sentiment classes in both sets. This was crucial to avoid training or evaluating the models on a skewed sample, which could lead to biased performance metrics. The split ratio used was 90% for training and 10% for testing, providing a substantial amount of data for model training while still retaining enough distinct data points for an accurate assessment of model generalizability.

#### Model Selection and Justification
Several models were selected based on their common usage in text classification tasks and their varying complexity, from simple probabilistic models to more sophisticated neural networks:

1. *Multinomial Naive Bayes (MNB):* Known for its effectiveness in dealing with text data due to its assumption of feature independence and its efficiency in training.
2. *Logistic Regression:* Provides a robust baseline with its capability to output probabilities for each class, useful for threshold tuning in class predictions.
3. *Dense Neural Network (DNN):* Utilizes layers of neurons with activation functions to capture non-linear relationships in the data.
4. *Simple RNN and GRU:* These sequential models are designed to capture temporal dependencies in text data, potentially offering superior performance on sequence-based data like text.

Each model was chosen to explore different aspects of model behavior—simplicity vs. complexity, linear vs. non-linear decision boundaries, and memory utilization in sequence processing.

#### Model Training and Hyperparameter Tuning
Models were trained using the preprocessed and vectorized text data. Hyperparameters such as learning rate, number of layers, and number of neurons in each layer were initially set based on common practices and fine-tuned iteratively based on model performance on the validation set. Techniques such as early stopping were employed to prevent overfitting, particularly for neural network models.

#### Validation and Metrics
The models were evaluated using a variety of metrics to assess their performance comprehensively:
- *Accuracy:* To measure the overall effectiveness of the models.
- *Precision, Recall, and F1-Score:* To understand model performance concerning each class, important in imbalanced datasets.
- *Confusion Matrix:* Provided insights into the types of errors made by the models (e.g., false positives and false negatives).

#### Overfitting/Underfitting Checks
Regular checks for overfitting or underfitting were conducted by comparing training and validation performance. If a model showed high training accuracy but low validation accuracy, it was an indication of overfitting. Conversely, if both training and validation accuracies were low, it pointed to underfitting. Appropriate measures, such as adjusting the model's complexity or training duration, were taken based on these findings.

#### Conclusion
This project focused on improving sentiment analysis by using machine learning models to classify user reviews. One of the key challenges was the imbalance in the dataset, particularly the lack of neutral reviews. To solve this, we generated 15,000 synthetic neutral reviews using predefined patterns and word pools. This approach helped balance the sentiment distribution, making the data more representative and improving model performance. Through various preprocessing techniques, such as cleaning and tokenizing the text, and balancing the dataset, we ensured the data was well-prepared for training.

The project also explored different machine learning models, ranging from simpler models like Multinomial Naive Bayes to more complex models like Neural Networks. The evaluation showed that incorporating the synthetic neutral reviews led to better and more stable model performance, especially in handling the imbalance in sentiment labels. By using various evaluation metrics, we were able to assess model performance comprehensively and fine-tune hyperparameters for optimal results. Ultimately, this approach improved the accuracy and fairness of the sentiment analysis system, showing that balancing datasets and using a mix of models can lead to better text classification results.
