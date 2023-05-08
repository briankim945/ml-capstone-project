# Judging a Book by it Cover

## Introduction

In this project, I attempt to solve the problem of applying sentiment analysis to book covers, using the cover images of different books to predict the emotional content of the text. Currently, senti- ment analysis is very focused on textual analysis, such as finding the emotional intent behind posts on social media. I believe that it is possible to expand sentiment analysis on images. Book covers provide an interesting new area to explore. Books, by nature, involve much more nuanced emotional cues than food reviews or social media posts, and book covers are designed to give potential readers a hint as to the contents, despite admonitions “not to judge a book by its cover.”

To create the dataset for this project, I draw on the book information provided by the Goodreads website. According to the company’s ‘About Us’ page, “Goodreads is the world’s largest site for readers and book recommendations.” [1] In particular, it collects information about books – including titles, covers, and reviews – and allows users to provide commentary on books for other users to see. I scrape the Goodreads website to build a database of book cover images and user-created labels that serve as emotional classifications. The resulting dataset has significant imbalances in the different emotional classes.

I primarily experiment with the ResNet-50 model, a specific type of convolutional neural network (CNN) introduced in the 2015 paper “Deep Residual Learning for Image Recognition” by He Kaiming, Zhang Xiangyu, Ren Shaoqing, and Sun Jian and implemented in the TensorFlow library. The bulk of my work is about testing out various approaches to making ResNet-50 more robust against the imbalances in my dataset.

## Repo Structure Outline

```
+
|- README.md # this file
|- BinaryToMulti.ipynb # Final file with two-part model (binary classification on SAD/NOT_SAD, then HAPPY/FEAR, with ResNet-50 + SMOTE + Focal Loss) training and testing
|- BinaryToMultiWithNorm.ipynb # Attempt for two-part model training and testing with normalization
|- ModelEvaluation.ipynb # Initial code for evaluating each model in aggregate and by emotional category
|- ModelEvaluation2.ipynb # Final code for evaluating each model in aggregate and by emotional category, with greater visualization
|- ModelTestingv1.ipynb # Initial round of model experimentation (ResNet-50, ResNet-50 with oversampling, ResNet-50 with SMOTE, ResNet-50 with undersampling, ResNet-50 with focal loss)
|- ModelTestingv2.ipynb # Further round of model experimentation (ResNet-50 with normalization, ResNet-50 with normalization and focal loss)
|- VGG.ipynb # Model experimentation with VGG-16 and VGG-16 with oversampling
|- VGG_terminal.py # Extra code for running VGG-16 in the event that Jupyter Lab is unable to allocate memory for testing purposes
|- md_images # Folder for images for README
|+ Data # Folder for various data collection and exploration
 |- EDA.ipynb # Exploratory Data Analysis, file for carrying out preliminary explorations of data and creating persistent train/test splits
 |- Scraping.ipynb # File for web scraping of Goodreads and retrieving relevant book data
```

## Work

First, I decided on the emotional categories that I would classify models into. I then collected data from Goodreads and separated the data into those categories. I explored the data, discovering major imbalances, and I experimented with the popular image-classifier CNNs ResNet-50 and VGG-16. I finally sought ways to mitigate the problems raised by the data imbalances, including undersampling, oversampling, and SMOTE.

### Categorizing Emotions
I decided to use the emotion categories defined by the psychologist Paul Ekman: Happiness, Sadness, Fear, Disgust, Anger and Surprise. [5] This seemed to be an intuitive division of books, particularly considering the different genres available. Tragedies fit naturally into Sadness, horror novels fit naturally into Fear, and suspenseful stories that heavily use twists and unexpected plot developments will probably be seen as Surprise.
    
### Collecting Data

I initially started by exploring popular datasets of books from a variety of sources, such as on Kaggle. However, I quickly found that most book data, including datasets built using Goodreads, included text data on the books – such as titles, descriptions, and ratings – but not cover images. This is probably to limit the size of datasets.

To get around this, I started with the Goodreads website, which provides large collections of books into categories based on popularity and genre. Since nearly every book came with a cover image on the website, I planned to use the Python libraries urlopen and BeautifulSoup to scrape book covers from the website.

I also needed to categorize the book images. To this end, I decided to use ‘shelves.’ Shelves are a feature of the Goodreads website where users can add simple descriptors to a book, such ‘fun’ or ‘scary.’ Each book has a shelves page, and by using the Python selenium library, I was able to scrape each book’s shelves in order of the number of users who had added that tag to each book.

To that end, I built a set of synonyms for each of the emotional categories, and then I used a simple Python function to find the first synonym I had flagged as important in each book’s shelves. Since these were ordered by the number of times they were used, I was able to get the most popular descriptor for each book which could be linked to a specific emotional category. The category which the synonym came from then became the label for the book. This created a dataset composed of book covers and emotional categories as labels.

For the initial scraping, I got my books from Goodreads’s ‘Best Books Ever’ List, since these seemed to be the most popular, and I thought that these might have the most user engagement and, thus, the most user tags. For the dataset, the inputs were only the book cover images - shrunk to identical dimensions - and the classifications labels were the most-used emotion labels.


### Data Exploration
After accounting for books that did not have usable shelves or broken image links, I had a dataset of 3289 samples. This seemed to be a promisingly large sample size. However, upon further exploration, I found a serious issue.

![Imbalance in favor of sadness](md_images/overall-bar.png "Bar graph showing number of samples per emotional category.")
<!-- \begin{figure}[H]
    \begin{center}
     \includegraphics[width=0.8\linewidth]{overall_bar.png}
     \caption{Bar graph showing number of samples per emotional category.}
    \end{center}
 % \label{fig:imgA}
\end{figure} -->
<!-- \noindent  -->
As the graph shows, there are serious imbalances in the number of samples for each emotional category. Books labeled as sad make up the majority of the dataset, books tagged with ‘fear’ make up a majority of the books left, and so on.

### Initial Approach
Based on the related works that I had read, I decided to start by applying ResNet-50 to the classification problem to see how well the model would be able to predict the labels of the book cover data without major modifications. I also decided to experiment with VGG-16, a much larger CNN, to see how larger model sizes would affect performance.

This led to the following results:
| | Test Accuracy	| Epoch Count | Time per Epoch |
| ---------------- | --------------- | --------------- | ------------ |
| ResNet-50 | 0.6200607902735562 | 25 |  3005 |
| VGG-16  | 0.6231003039513677 | 25 & 1180 |

I found little difference between the two models in overall accuracy. In addition, when looking at confusion matrices of the results:

![ResNet-50 default results](md_images/overall-bar.png "Bar graph showing number of samples per emotional category.")
\begin{figure}[H]
    \begin{center}
     \includegraphics[width=0.8\linewidth]{resnet1.png}
     \caption{Confusion matrix of ResNet-50 outputs.}
     \includegraphics[width=0.8\linewidth]{vgg1.png}
     \caption{Confusion matrix of VGG-16 outputs.}
    \end{center}
 % \label{fig:imgA}
\end{figure}

\noindent It seemed clear that the Sadness label was being selected too frequently.

 
\subsection{Responding to Data Imbalances}
I first approached the problem of resolving the data imbalances by applying undersampling, oversampling, and the Synthetic Minority Oversampling Technique (SMOTE). Undersampling is a technique to balance data by keeping all samples in minority classes and getting rid of samples in large classes. Oversampling seeks to do the opposite by introducing more samples to the minority classes. In my case, I randomly sampled from the existing minority classes with replacement to get a more even distribution of data. Finally, SMOTE is a specialized oversampling technique synthesizes new samples from existing samples, effectively creating more data points based on the old ones.
\newline
\newline
In addition to trying to fix problems in the data, I tried to tailor the models to be more responsive to the differences in the data distribution. In my case, I applied focal loss – particularly Focal Cross Entropy Loss – to get the model to weight incorrect predictions for minority classes more heavily.
\newline
\newline
Finally, I tried to alter the plan more thoroughly in response to the problems I ran into. Based on the results of my previous experimentation, I created a more complex prediction process. I first turned the problem from a multiclass classification into a binary classification problem, only predicting if a book was tagged as sad or not sad. For the books tagged as not sad, I then created another model to predict if they were classified fear or happiness. I dropped disgust and anger entirely, since there were so few in the dataset. This created a two-model pipeline for predicting whether a book is sad, happy, or scary.

### ResNet-50

#### Standard:
Loss = 3.9193613529205322
Test Accuracy = 0.628953754901886

#### Oversampling
Loss = 2.284522533416748
Test Accuracy = 0.57351154088974

#### Undersampling
Loss = 1.7097039222717285
Test Accuracy = 0.019441070035099983

#### SMOTE
Loss = 2.6277992725372314
Test Accuracy = 0.6379100680351257

#### Focal Loss (SigmoidFocalCrossEntropy)
Loss = 0.181303933262825
Test Accuracy = 0.6767922043800354

#### Normalized
Loss = 3.4941563606262207
Test Accuracy = 0.6211936473846436

#### Normalized with Focal Loss (SigmoidFocalCrossEntropy)
Loss = 0.24491138756275177
Test Accuracy = 0.6504263281822205

### VGG16

#### Standard
Loss = 2.4211628437042236
Test Accuracy = 0.6184689998626709

#### Oversampling
Loss = 5.077214241027832
Test Accuracy = 0.6184689998626709
