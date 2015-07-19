# machine
Assignment Coursera Project writeup

Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The data was collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways ( "classe" ):  
                A - Regular  
                B - Throwing the elbows to the front   
                C - Lifting the dumbbell only halfway    
                D - Lowering the dumbbell only halfway  
                E - Throwing the hips to the front 

The goal of this project is to predict the manner in which the participants did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

The training and test data for this project are available on: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

QUESTIONS ADDRESSING

HOW WAS BUILT THE MODEL ( Chosen Regressions method ) ?

        I chose the two most common method to make this analysis: The CART (Classification and regression Tree) and The Random Forest. The CART method is the standard machine learning technique in ensemble terms, corresponds to a weak learner. In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets.The Random Forest (Breiman, 2001) is an ensemble approach which is a collection of "weak learners" that can come together to form a "strong learner".  
        - CARET method is easy to interpret, fast to run but hard to estimate the uncertainty    
        - RANDOM FOREST has a very good accuracy, but difficult to interpret and low speed  to run (may take a couple of hours)

HOW WAS CHOSEN THE CROSS VALIDATION?

        There are several methods to estimate the model accuracy among them, Data Split, Boostrap, K-Fold Cross validation, Repeated K-Fold Cross validation and Leave One Out Cross validation. I choose the DATA SPLIT due to the very large amount of data of thsi problem.   
        - Data Splitting: Involves partitioning the data into an explicit Training dataset used to prepare the model, (typically 70% ) and an unseen Test dataset ( typically 30% )used to evaluate the models performance on unseen data. It is useful for a very large dataset so that the test dataset can provide a meaningful estimation of the performance or for when you are using slow methods and need a quick approximation of performance     
        - K-Fold Cross Validation; Involves splitting the dataset into K-subsets. For each subset is held out while the model is trained on all other subsets. This process is completed until accuracy is determine for each instance in the dataset, and an overall accuracy estimate is provided.     
        - Leave One Out Cross Validation: a data instance is left out and a model constructed on all other data instances in the training set . This is repeated for all data instances.


WHAT YOU THINK THE EXPECTED OUT OF SAMPLE ERROR IS ( = 1 - ACCURACY )?

        The two methods chosen provided diferente accuracy. the CART Method provided an accuracy of 60,65%, consequently the expected out of the sample error 1- 60,65% = 39,35% and on the other hand the RANDOM FOREST method provided a much better accuracy 99,32%, giving an expected out of sample error of 1 - 99,32% = 0.68%.


USE THE FINAL PREDICT MODEL TO PREDICT 20 DIFFERENT TEST CASES.   

        Answers submitted in the project submission








