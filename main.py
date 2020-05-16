# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
import time
import warnings
import pickle


# Starts time to check how long the program runs for
start_program_time = time.time()

# Ignores the warnings when training models
warnings.filterwarnings('ignore')

# Initialize variables
categorical_data = []
numeric_data = []
categorical_test_data = []
numeric_test_data = []
binary_values = {'N': 0, 'Y': 1}


"""Below is preprocessing all the training data"""

# Read file
train_file = pd.read_csv('train.csv')

# Gets rid of loan id's from data
train_file.drop('Loan_ID', axis=1, inplace=True)

# Categorize columns by categorical or numerical data
for i, c in enumerate(train_file.dtypes):
    if c == object:
        categorical_data.append(train_file.iloc[:, i])
    else:
        numeric_data.append(train_file.iloc[:, i])

# Transposes categorical and numeric data
categorical_data = pd.DataFrame(categorical_data).transpose()
numeric_data = pd.DataFrame(numeric_data).transpose()

# Fill in missing values
## Categorical: filled in by most occurring value
## Numerical: filled in with bfill method: value right above
categorical_data = categorical_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
numeric_data.fillna(method='bfill', inplace=True)

# LabelEncoder helps transform the data
le = LabelEncoder()

# Predictor variable
predict = categorical_data['Loan_Status']

# Gets rid of predictor variable from data
categorical_data.drop('Loan_Status', axis=1, inplace=True)

# Transforms predictor variable into numbers
predict = predict.map(binary_values)

# Transform categorical variables into numbers
for i in categorical_data:
    categorical_data[i] = le.fit_transform(categorical_data[i])

# Adds changes to the training file
train_file = pd.concat([categorical_data, numeric_data, predict], axis=1)

# Separate two variables into training data and predictor variable
X = pd.concat([categorical_data, numeric_data], axis=1)
y = predict

# Splits into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)


# Graphs the correlation between all variables in heat map
plt.title('Variable Correlations')
sns.heatmap(train_file.corr(), annot=True)
# plt.show()  Get rid of hashtag if you want to show the graph above


"""Training models start from below"""

# Different models
dict_of_models = {
    'Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(random_state=46),
    'K Neighbors Classifier': KNeighborsClassifier(),
    'SVC': SVC(kernel="rbf", C=5, random_state=24),
    'Decision Tree Classifier': DecisionTreeClassifier(max_depth=1, random_state=42)
}


# Function that converts dictionary keys and values into their own respective lists
def get_name_list(dictionary):
    list_keys = []
    list_values = []
    for key in dictionary.keys():
        list_keys.append(key)
    for value in dictionary.values():
        list_values.append(value)
    return list_keys, list_values


# Pulls list of names of models and models from dictionary
names_of_models, list_of_models = get_name_list(dict_of_models)


# Loss scores and prints out accuracy
def loss(y_true, y_pred, retu=False):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_one = f1_score(y_true, y_pred)
    log = log_loss(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    if retu:
        return precision, recall, f_one, log, accuracy
    else:
        print("Accuracy: ", accuracy)


# Trains all the models
def train_models(names, models, loops=1):
    # Variables from global scope
    global x_train, x_test, y_train, y_test

    # Initialize variables
    best_accuracy = 0
    best_model = ""
    num_loop = 0

    # Runs all the models for the user inputted number of loops
    for x in range(loops):
        print("Running Loop #{0}".format(x))

        # Splits test and training data
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        # Runs through all the models in the dictionary defined earlier
        for i in range(len(models)):
            print(names[i] + ":")

            # Trains the model with the training data
            ml_model = models[i].fit(x_train, y_train)

            # Gets accuracy of the model on the test data
            model_accuracy = ml_model.score(x_test, y_test)
            print("Accuracy: ", model_accuracy, "\n")

            # Keeps track of the best model, what the accuracy was, and which loop it was
            if model_accuracy > best_accuracy:
                best_accuracy = model_accuracy
                best_model = names[i]
                num_loop = x + 1
                with open("loan_model.sav", "wb") as file:
                    pickle.dump(ml_model, file)

    print("The best model was {0} in loop #{1} and the accuracy was {2}".format(best_model, num_loop, best_accuracy))


# Starts second time to check how long training the model takes
start_model_time = time.time()

# Calls function and trains all the models to discover the best model
train_models(names_of_models, list_of_models, loops=10000)

# Prints out the time it took to execute the whole program
print("The models took: {:.2f} seconds to train".format(time.time() - start_model_time))


# Displays a heat map of correlations between every variable
data_correlation = pd.concat([x_train, y_train], axis=1)
correlation = data_correlation.corr()
plt.figure(figsize=(20,14))
plt.title('Variable Correlations')
sns.heatmap(correlation, annot=True)
# plt.show()  Get rid of hashtag if you want to show the graph above


"""Below uses the best trained model and predicts the test data file"""

# Open and load training model
loan_model = open("loan_model.sav", "rb")
loan_model = pickle.load(loan_model)

# Read file
test_file = pd.read_csv('test.csv')

# Gets rid of loan id's from data
test_file.drop('Loan_ID', axis=1, inplace=True)

# Categorize columns by categorical or numerical data
for i, c in enumerate(test_file.dtypes):
    if c == object:
        categorical_test_data.append(test_file.iloc[:, i])
    else:
        numeric_test_data.append(test_file.iloc[:, i])

# Transposes categorical and numeric data
categorical_test_data = pd.DataFrame(categorical_test_data).transpose()
numeric_test_data = pd.DataFrame(numeric_test_data).transpose()

# Fill in missing values
## Categorical: filled in by most occurring value
## Numerical: filled in with bfill method: value right above
categorical_test_data = categorical_test_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
numeric_test_data.fillna(method='bfill', inplace=True)

# Transform categorical variables into numbers
for i in categorical_test_data:
    categorical_test_data[i] = le.fit_transform(categorical_test_data[i])

# Add data to file
test_file = pd.concat([categorical_test_data, numeric_test_data], axis=1)

# Adds column with predicted values to test file
test_file["Loan_Status"] = loan_model.predict(test_file)

# Makes the changes to the file
test_file.to_csv("test.csv")

# Prints all the predicted values
print(test_file["Loan_Status"])


"""Below are test code for some training models"""

"""
# Test to see best logistic regression model
running_best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    logistic = linear_model.LogisticRegression(random_state=46)
    logistic.fit(x_train, y_train)
    acc = logistic.score(x_test, y_test)
    print(acc)
    if acc > running_best:
        running_best = acc
"""


"""
# Test to see best SVC model
running_best = 0
loop = 0
for i in range(20, 35):
    loop += 1
    print("This is the {} loop".format(loop))
    for j in range(2, 6):
        for _ in range(3):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
            svm_model = svm.SVC(kernel="poly", C=j, random_state=i)
            acc = svm_model.fit(x_train, y_train).score(x_test, y_test)
            if acc > running_best:
                running_best = acc
                j_save = j
                i_save = i

print('best', running_best)
print('j', j_save)
print('i', i_save)
"""

"""
# Test to see average regression model scores to see which variable to drop
categorical_data.drop('Self_Employed', axis=1, inplace=True)
categorical_data.drop('Dependents', axis=1, inplace=True)
categorical_data.drop('Education', axis=1, inplace=True)
total = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    logistic = LogisticRegression(random_state=46)
    logistic.fit(x_train, y_train)
    acc = logistic.score(x_test, y_test)
    total += acc

print(total / 1000)
print("time elapsed: {:.2f}s".format(time.time() - start_time))
"""


# Prints out the time it took to execute the whole program
print("The program took: {:.2f} seconds to execute".format(time.time() - start_program_time))
