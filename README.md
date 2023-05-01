# Judging a Book by it Cover

## Overview

This project aims to apply sentiment analysis to images, focusing on book covers. My thinking is that book covers, because they need to convey information about the emotional content of the book they are advertising, provide an interesting source of images that can be used to train sentiment analysis.

I scraped my data from the GoodReads website, including both book cover images and relevant data about the book - user tags, summaries, titles, etc. - and used this data to generate input data - the images - and classifications for each data point - an approximation of the emotional label for each point.

I trained models on the data using popular image classification models, primarily ResNet-50 and VGG. I quickly found problems with training due to imbalances in the data, and I experimented with a number of different approaches to working around this imbalance. Primarily, I explored undersampling, oversampling, and SMOTE.

My final approach was to divide the classification process into two steps. First, I turned the problem into a binary classification problem, with data points being labeled as "SADNESS" or "NOT_SADNESS", since sad books were the largest subgroup of books and were heavily skewing the data. After that, I applied a multiclass model to the "NOT_SADNESS" images into "HAPPINESS", "SURPRISE", and "ANGER".

STEP 1:

Data Collection

Possible starting points:
https://www.goodreads.com/list/show/1.Best_Books_Ever (4480 samples)

https://www.goodreads.com/shelf/show/best-covers (6859 samples)

Genres
Descriptors (happy, sad, exciting, slow)
Look at "Top Shelves"

## Preliminary Results

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