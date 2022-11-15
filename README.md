# CMPE255-Team12

## Team 12

## Abstract

In 2021, the recruiting industry was a 136 billion dollar industry1 in the United States alone. However, sorting the most appropriate job for an applicant through a resume is a difficult process for both the applicant as well as the employer. This process is often manual and time consuming. In order to reduce the manual time and effort involved to properly classify an applicant, we propose using a BERT transformer to classify resumes. We plan on feeding cleaned labeled resume data into a BERT transformer in order to train the model to classify resumes. We anticipate that there will be difficulties with the time that it takes to run the model as well as accuracy issues. In order to mitigate these problems, we will try out different BERT models such as: BERT-base, Small BERT, ALBERT, Electra, BERT Experts, and BERT with Talking-Heads Attention. 
In doing so, we hope to not only provide better automated classification of resumes but    recommend potential suggested classifications based on the BERT model. Based on previous BERT attempts with other classification problems such as comment sentiment analysis. We anticipate that we will be able to achieve around 80% accuracy with our resume classification attempts. While this is far from perfect, this method requires zero manual effort and would serve as a great starting point for the resume classification process. We hope this will greatly improve the efficiency of the resume classification process for applicants and employers.

## Intro

Recruiting suitable candidates for the job role is time-consuming but an important task. The number of applicants in the job market can be overwhelming. Especially with the different types of job roles existing along with the increasing number of applications from candidates. Resume classification is needed to make the process of selecting the appropriate candidates for the role easier. This tool will help recruiters identify suitable candidates based on their skill set. Applicant tracking systems “help companies in the process of recruiting new professional figures or re-assigning resources [2].” They are extensively used in the recruitment process to find candidates with the required qualifications. However, applicant tracking systems require a manual evaluation that is time-consuming. Recruiters often need to ensure that the resume isn’t manipulated using keywords. This results in inefficiency in the recruitment process of selecting the appropriate candidates for the role.

This paper presents BERT to classify resumes. BERT is a language representation model that uses bidirectional representation and can create models for different processing tasks. We will use BERT for our resume-job classification project. BERT uses a masked language model (MLM) to “combine left and right context in all layers to enable a bidirectional Transformer [3].” To process our model using BERT, we will use the pre-training and fine-tuning process. Fine-tuning BERT is inexpensive compared to pre-training. We will first pre-train our model on labeled data while fine-tuning our model to be initialized with the pre-trained data. This allows us to capture the language modeling of the resume. The span of texts such as skills on a resume is differentiated by their special token and the learned embedding to show where the token belongs. BERT shows significant improvement compared to other systems.

The structure of the paper is as follows. Section 2 presents the summary of the prior research on other models of resume classification. Section 3 presents the proposed methodologies used in our project. Section 4 discusses our resume classification results and the analysis using BERT. Finally, Section 5 concludes the paper and discusses future work.


## Literature Review

Introduced in 1986 by Rumelhart [4], the recurrent neural network (RNN) is one of the earliest and the most popular architectures in dealing with sequential data inputs. According to Schmidt (2019) [5], as compared to non-recurrent feedforward networks which pass information through the network without cycles, the RNN has cycles and transmits information back into itself in order to detect patterns in sequential data.

However, in the paper “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network” (2020) [6], Sherstinsky demonstrated that RNNs suffer from issues known as “vanishing gradients” and “exploding gradients” during training since the recurrent weights would get amplified as the size of the input grows. This would lead to unsatisfactory accuracy when processing larger corpora, including resumes.

One solution to address the vanishing or exploding gradients problem is a gradient-based method known as Long Short-Term Memory (LSTM) introduced by Hochreiter and Schmidhuber (1997) [7]. According to Hochreiter, LSTMs have a chain-like structure similar to standard RNNs, but the repeating pattern is more complicated, which includes special units such as memory cells, input gates, output gates, and forget gates. The memory cell remembers value in the long term, and the gate units are conditionally activated in order to regulate the flow of information into and out of the cell. This mechanism allows the recurrent gradients to flow unchanged.

## Solutions (Methodology)

We propose using small BERT initially for computational resource limitations as small BERT is the fastest and smallest model. For the first set, we are going to use the first 128 words in the resume for training time purposes. We will extend the text length to improve accuracy. We will have the following classes of resumes: 'Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing', 'Mechanical Engineer', 'Sales', 'Health and fitness', 'Civil Engineer', 'Java Developer', 'Business Analyst', 'SAP Developer', 'Automation Testing', 'Electrical Engineering', 'Operations Manager', 'Python Developer', 'DevOps Engineer', 'Network Security Engineer', 'PMO', 'Database', 'Hadoop', 'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing’. We want to do a validation set of 20% of our total data and use 80% of the total data as training data. With 963 total resumes, this comes out to be a training set of 770 resumes and a validation set of 193 resumes. We will use Binary Cross Entropy to calculate our loss through 5 epochs.

## Reference

[1] https://www.statista.com/statistics/873648/us-staffing-industry-market-size/

[2] A. Bondielli and F. Marcelloni, “On the use of summarization and transformer architectures for profiling résumés,” Expert Systems with Applications, vol. 184, p. 115521, 2021.

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional Transformers for language understanding,” arXiv.org, 24-May-2019. [Online]. Available: https://arxiv.org/abs/1810.04805. [Accessed: 16-Oct-2022].

[4] Rumelhart, D., Hinton, G. & Williams, “R. Learning representations by back-propagating errors”. Nature 323, 533–536 (1986). https://doi.org/10.1038/323533a0

[5] Robin M. Schmidt, “Recurrent Neural Networks (RNNs): A gentle Introduction and Overview,” 23-Nov-2019.

[6] Alex Sherstinsky, “Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network,” 2020.

[7] Sepp Hochreiter, Jürgen Schmidhuber, “Long Short-Term Memory,” Neural Comput 1997; 9 (8): 1735–1780. doi: https://doi.org/10.1162/neco.1997.9.8.1735
