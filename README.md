# Resume Classification

### Team 12

## Abstract

In 2021, the recruiting industry was a 136 billion dollar industry1 in the United States alone.
However, sorting the most appropriate job for an applicant through a resume is a difficult process
for both the applicant as well as the employer. This process is often manual and time consuming.
In order to reduce the manual time and effort involved to properly classify an applicant, we propose
encoding the resume with a NLP method and then classifying the resumes with a classification
algorithm. We plan on feeding cleaned labeled resume data into a language model in order to train
the model to classify resumes. While this is far from perfect, this method requires zero manual
effort and would serve as a great starting point for the resume classification process. We hope this
will greatly improve the efficiency of the resume classification process for applicants and employers.
We anticipate that there will be difficulties with the time that it takes to run the model as well as
accuracy issues. In order to mitigate these problems, we will try out a combination of encoding
models and classification algorithms. We will look at a wide range of models and algorithms,
both old and new and compare their performance. In doing so, we hope to not only provide
better automated classification of resumes but to also get an understanding of the improvement of
language classifications and why newer techniques such as transformers have dominated the space.

## Intro

Recruiting suitable candidates for the job role is time-consuming but an important task. The number of applicants in the job market can be overwhelming. Especially with the different types of job roles existing along with the increasing number of applications from candidates. Resume classification is needed to make the process of selecting the appropriate candidates for the role easier. Resume classification will help recruiters identify suitable candidates based on their skill set according to the job descriptions. Other tools such as applicant tracking systems “help companies in the process of recruiting new professional figures or re-assigning resources [2].” They are extensively used in the recruitment process to find candidates with the required qualifications. However, applicant tracking systems require a manual evaluation that is time-consuming. Recruiters often need to ensure that the resume isn’t manipulated using keywords. This results in inefficiency in the recruitment process of selecting the appropriate candidates for the role. There is a need to categorize resumes based on the job descriptions. It would help separate the relevant resumes that the recruiters are looking for from the large amount of resumes that may not have the necessary qualifications.

To overcome the issues and inefficiency of the recruitment process, this paper presents TF-IDF + logistic regression, TF-IDF + random forest, TF-IDF + multinomial Naive Bayes, LSTM, BERT, GPT-3 + logistic regression, and GPT-3 + random forest as potential classification pipelines for resumes. We survey these models to find the best algorithm to classify resumes. To extract the relevant information, we propose word embeddings such as TF-IDF. TF-IDF captures both relevance and frequency of words such as relevant skills on the resume. We also propose GPT-3 as word embeddings to extract similarities of relevant information. We propose supervised learning classification models such as  Naive Bayes, Random Forest, and Logistic Regression to predict the category of resumes. Naive Bayes is a probabilistic classifier based on Baye’s theorem. Random Forest uses decision trees to predict the accuracy. Logistic regression finds the correlation between relevant information and the category of resumes based on the logistic function. Long short-term memory (LSTM) was before transformers were developed. It uses feedback connections and memorizes relevant information from the resume to classify sequentially. However, it suffers from long sequences of text. We propose BERT to handle the limitations of LSTM. BERT is capable of handling long sequences of text. The goal of this paper is to find the best suitable algorithm for classifying resumes using one of these models.


## Related Work

Introduced in 1986 by Rumelhart [4], the recurrent neural network (RNN) is one of the earliest and the most popular architectures in dealing with sequential data inputs. According to Schmidt (2019) [5], as compared to non-recurrent feedforward networks which pass information through the network without cycles, the RNN has cycles and transmits information back into itself in order to detect patterns in sequential data.

However, in the paper “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network” (2020) [6], Sherstinsky demonstrated that RNNs suffer from issues known as “vanishing gradients” and “exploding gradients” during training since the recurrent weights would get amplified as the size of the input grows. This would lead to unsatisfactory accuracy when processing larger corpora, including resumes.

One solution to address the vanishing or exploding gradients problem is a gradient-based method known as Long Short-Term Memory (LSTM) introduced by Hochreiter and Schmidhuber (1997) [7]. According to Hochreiter, LSTMs have a chain-like structure similar to standard RNNs, but the repeating pattern is more complicated, which includes special units such as memory cells, input gates, output gates, and forget gates. The memory cell remembers value in the long term, and the gate units are conditionally activated in order to regulate the flow of information into and out of the cell. This mechanism allows the recurrent gradients to flow unchanged.

Different models have been proposed to classify resumes. Roy et al. Pal et al. [12] proposed a resume classification using various Machine Algorithms such as Naive bayes, Support Vector Machine (SVM) and Random Forsest. Their preprocessing of data consisted of proper stemming and lemmatization. TF-IDF Vectorization was used to extract the relevant information from the resume. Classification algorithms such as Naive bayes, Support Vector Machine (SVM) and Random Forsest are used for comparison. Overall, Random Forest classification model gave the best accuracy for predicting resumes in their test data. However, their model for Random Forest only resulted in a 70\% accuracy. Roy et al. [8] proposed an approach for resume classification by using k-NN to categorize resume that are the nearest to job descriptions using different classifiers. The authors used TF-IDF for feature extraction after the data was preprocessed. To categorize the resume into their proper categories, the authors used linear support vector classifier as it provided the best accuracy. The resumes that are the closest to the provided job descriptions are identified using k-NN. Overall, the model LinearSVM classifier gave them the best accuracy with 78.53\%. It also helps in assisting recruiters of getting recommendations for the job application using k-NN. Bondielli et. al [2] proposed resume classification using the summarization technique and transformer architectures to classify resumes. The summarization was used successfully to condense the texts of resumes and remove the redundancy. The authors used BM25-TextRank algorithm to be more efficient in summarization and transformer-based architectures such as BERT. Hierarchical clustering algorithms was used on resume embeddings to give the best recommendation of resumes with relevant information. Barducci et. al [9] proposed an end-to-end framework to classify resumes by extracting only the relevant information based on skills or work experience using NLP and ML techniques. The researchers used segment extraction technique to extract and represent resumes based on different information. The researchers then used Named Entity Recognition (NER) to extract relevant features from each segment of the resume. An embedding model such as BERT was used in order to perform the NER task. Static embedding such as FastText gave worse performance compared to contextual embedding such as BERT. Relevant keywords based on skills or work experience are extracted since they contain the most important features of the resume. Jiechieu and Tsopze [11] proposed a multi-label resume classification using Convolutional Neural Networks (CNN) to predict the category of resumes. CNN is capable of distinguising relevant features from low-level features such as the relevant skills from the resume using convolutional filters. The researchers extracted skills from resume as multi-label classification. The researchers used word embeddings to transform the text-resume into a matrix representation as an input for CNN for preprocessing. CNN is then used as multi-label classifier to filter the relevant information to each category of the resume. The CNN model showed excellent accuracy in predicting the category of each resumes. It achieved a 99\% accuracy to predict resumes in their experiment.

## Project Description

In order to understand the improvements to language classification models we first need to survey the improvements. Unfortunately, the major of the improvements in language models has been to the encoding layer, specifically the addition of transformers. As such, we will compare TF-IDF, an older encoding technique, to the two most prominent transformer model encoding techniques, BERT and GPT-3. To add complexity to this, we will also be looking at different algorithms (logistic, random forest, and multinomial naives bayes against TF-IDF and transformers. We hope to answer 2 separate questions:

1. What are improvements, pros, and cons of the newer transformer based encoding models as
compared to traditional models such as TF-IDF?

2. What is the most suitable classification model?

In our project, we will be classifying the 25 different categories of resumes from the kaggle dataset
(https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset). We would be using the
above list of language encoding models with their respective classification models in order to run the
classifications and compare techniques. We will be using accuracy against a holdset test set in order
to compare the models to each other. Additionally, we will qualitatively look at the errors in order to
gleam some insight into the problem.

## Experimental Methodology

### Data Preprocessing
We used pandas and numpy to load our data set from UpdatedResumeDataSet.csv. We then used
numpy and pandas to create a stratified split of our data. Since we have 25 different tagged classes,
we can not just randomly sample as we would undersample certain categories. For BERT, in order
to create a keras dataset. We needed to use python to extract the sample resumes into text files for
processing. We wrote a data parser for this.
For cleaning the data, we used regular expressions “re.sub(r”[ˆa-zA-Z0-9]+”, ’ ’, clean)” to remove
all of the special characters in the data. We also had to label encode all of the categories for the
resumes for preprocessing. The least amount of any particular category of resume is 20 and the most
is 84. This is a 4x difference, but still reasonable.
### Data Mining Models
#### LSTM
LSTM model was selected to demonstrate its limitation with long sequences of text for classification.
LSTM memorizes and finds the important information of the resume. Although LSTM fixes RNN’s
vanishing gradient problem, it still does not perform well when processing longer sequences.
#### TF-IDF + Random Forest
TF-IDF stands for Term Frequency–Inverse Document Frequency, which is a word embedding tech-
nique. It was designed to reflect how important a word is to a corpus, which matches the nature of
resume classification. Resumes for different jobs would include different collection of terms, and those
3
terms are usually strong predictors of the category of the resume. For example, the term “Java” is
usually found in resumes for Java Developers. With TF-IDF as a suitable embedding, we used Random
Forest Classifier. Diagram of the model architecture:
#### TF-IDF + Logistic Regression
TF-IDF was also used to extract the relevant information from the resume before Logistic regression
classifier is used. Logistic regression is used in finding correlations between variables. It uses a logistic
function to do the classification task by using the weighted combination of the input features. The
extracted features from TF-IDF is fed into Logistic regression model to predict the category of the
resume.
#### BERT
BERT was selected as a language classification model because it is state of the art, easy and readily
available for import, runs relatively quickly, used in industry, has a lot of documentation, and has
great classification accuracy. Due to these criteria we suspect that BERT can be used in a real life
scenario for resume classification. We used BERT as our language model and added 25 neurons for
the 25 different categories as the classifier using tensorflow keras. Between BERT and the 25 neuron
classifier, we will have a dropout layer in order to prevent model overfitting. Scikit learn was used to
compute the accuracy for the models. Diagram of the model architecture:
#### GPT-3 Embedding Babbage-001 + AdaBoost Regressor
We choose GPT-3 Zero Shoot as a classification model for all of the same reasons as BERT (state
of the art, easy and readily available for import, runs relatively quickly, used in industry, has a lot
of documentation, and has great classification accuracy.) GPT-3 was used via the OpenAI driver via
python. Pandas and Numpy were used for data preprocessing. The embedding generation is shared
between the two GPT-3 models. AdaBoost was chosen as a classifier because we wanted to have a
model that uses a boost classifier. Boost classifiers such as XGBoost and CatBoost often wins kaggle
competitions. We are using AdaBoost as our example of a boost classifier. AdaBoost tried to ensemble
weaker classifications into a stronger ensemble classification to improve results.
#### GPT-3 Embedding Babbage-001 + Random Forest
We also explored GPT-3’s similarity embedding for classification. In addition to GPT-3’s powerful
zero-shot text completion endpoint, it also has a variety of embedding models with distinct features.
We choose babbage-001 as our embedding model for its ability to capture text features, which is best-
suited for resume classification tasks. We then utilized Random Forest as the classifier after feature
embedding.

## Reference

[1] https://www.statista.com/statistics/873648/us-staffing-industry-market-size/

[2] A. Bondielli and F. Marcelloni, “On the use of summarization and transformer architectures for profiling résumés,” Expert Systems with Applications, vol. 184, p. 115521, 2021.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional Transformers for language understanding,” arXiv.org, 24-May-2019. [Online]. Available: https://arxiv.org/abs/1810.04805. [Accessed: 16-Oct-2022].

[4] Rumelhart, D., Hinton, G. & Williams, “R. Learning representations by back-propagating errors”. Nature 323, 533–536 (1986). https://doi.org/10.1038/323533a0

[5] Robin M. Schmidt, “Recurrent Neural Networks (RNNs): A gentle Introduction and Overview,” 23-Nov-2019.

[6] Alex Sherstinsky, “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network,” 2020.

[7] Sepp Hochreiter, Jürgen Schmidhuber, “Long Short-Term Memory,” Neural Comput 1997; 9 (8): 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735
