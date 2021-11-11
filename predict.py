import numpy as np
import pickle as p
import numpy as np
import inquirer

class Classifier:
    """ Functions to train and use a binary decision tree classifier with targets >50K and <=50K. 
        For training, all features described in the 1994 Adult Census dataset must be present in the file provided, and their order must not change. 
    """
    
    def __init__(self):
        self.encoders = []      # stores the encoders so that all inputs to the model are encoded the same way
        self.clf = None         # stores the trained model for use in future predictions
        self.features = []


    def useEncoders(self, X):
        """ Uses saved encoders to convert features to appropriate format for the model.
        Since the scikit-learn classification model that we are using takes continuous numeric features, 
        we must convert categorical features to their equivalent one-hot encoding.
        The input features X must not contain any categories that were not seen in the data that the saved encoders were trained on.
        
        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        oneHotX : numpy.ndarray
        """
        
        i = 0
        oneHotX = []
        for column in X.T:
            column = column.reshape(-1, 1)
            try:
                column.astype(float)
            except ValueError:
                enc = self.encoders[i]
                column  = enc.transform(column).toarray()
                i += 1
            if len(oneHotX) == 0:
                oneHotX = column
            else:
                oneHotX = np.concatenate((column,oneHotX),axis=1)

        return oneHotX

    def predictEncoded(self,X):
        """ Uses the saved trained model to make a prediction on input features X.
        X must already be encoded, and contain the same number of features as the saved model was trained on.
        A prediction of 1 corresponds to an income of <=50K, and a value of 0 corresponds to an income of >50K.
        
        Parameters
        ----------
        X : numpy.ndarray
        Y : numpy.ndarray
        
        Returns
        -------
        predictions : numpy.ndarray
        """

        predictions = self.clf.predict(X)
        return predictions
        
    def predict(self, X):
        """ Uses the saved trained model to make a prediction on input features X.
        X's features must be in their original format (as extracted from the csv, not encoded) 
        X also must contain the same number of features as the saved model was trained on.
        A prediction of 1 corresponds to an income of <=50K, and a value of 0 corresponds to an income of >50K.
        
        Parameters
        ----------
        X : numpy.ndarray
        Y : numpy.ndarray
        
        Returns
        -------
        predictions : numpy.ndarray
        """
            
        encodedX = self.useEncoders(X)
        predictions = self.predictEncoded(encodedX)
        return predictions

# Gather user input for the features 
capital_loss = [
  inquirer.Text('capital_loss', message="Enter capital loss (dollars): "),
]
capital_gain = [
  inquirer.Text('capital_gain', message="Enter capital gain (dollars): "),
]
hours_per_week = [
  inquirer.Text('hours_per_week', message="Enter the number of hours worked per week: "),
]
age = [
  inquirer.Text('age', message="Enter age (years): "),
]
fnlwgt = [
  inquirer.Text('fnlwgt', message="Enter final weight (number of people with statistics similar to this entry): "),
]
education_num = [
  inquirer.Text('fnlwgt', message="Enter number of years of education: "),
]
education = [
  inquirer.List('education',
                message="Choose highest education level attained",
                choices=['Doctorate','Masters','Bachelors', 'Prof-school', 'Some-college', 'Assoc-acdm', 'Assoc-voc','HS-grad', '12th','11th', '10th', '9th', '7th-8th', '5th-6th',  '1st-4th', 'Preschool'],
            ),
]
# education = inquirer.prompt(educationLevels)["education"]
marital_status = [
  inquirer.List('marital_status',
                message="Choose marital status",
                choices=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
            ),
]
# mStatus = inquirer.prompt(marital_status)["marital_status"]

occupation = [
  inquirer.List('occupation',
                message="Choose occupation",
                choices=['Tech­-support', 'Craft­-repair', 'Other­-service', 'Sales', 'Exec­-managerial', 'Prof-­specialty', 'Handlers-­cleaners', 'Machine­-op­-inspct', 'Adm­-clerical', 'Farming­-fishing', 'Transport­-moving', 'Priv­house­-serv', 'Protective­-serv', 'Armed­-Forces'],
            ),
]
workclass = [
  inquirer.List('workclass',
                message="Choose workclass",
                choices=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
            ),
]
# occupation = inquirer.prompt(occupations)["occupation"]
race = [
  inquirer.List('race',
                message="Choose race",
                choices=['White', 'Asian-­Pac­-Islander', 'Amer­-Indian­-Eskimo', 'Other', 'Black'],
            ),
]
# race = inquirer.prompt(races)["race"]
sex = [
  inquirer.List('sex',
                message="Choose biological sex",
                choices=['Female', 'Male'],
            ),
]
relationship = [
  inquirer.List('relationship',
                message="Choose relationship",
                choices=['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
            ),
]

native_country = [
  inquirer.List('native_country',
                message="Choose native country",
                choices=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
            ),
]


# Load trained model
modelfile = 'models/trained_model.pickle'
model = p.load(open(modelfile, 'rb'))

possibleQuestions = [age, workclass, fnlwgt, education, education_num, marital_status,occupation,relationship, race, sex, capital_gain,capital_loss,hours_per_week ,native_country]
possibleFeatures = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status","occupation","relationship", "race", "sex", "capital_gain","capital_loss","hours_per_week" ,"native_country"]
answers = []
for feature in model.features:
    questionIndex = possibleFeatures.index(feature)
    answer = inquirer.prompt(possibleQuestions[questionIndex])[feature]
    answers.append(answer)

# # Convert user data features into input array
answers = np.array(answers)

# Predict using trained encoders, model. A prediction of 1 corresponds to an income of <=50K, and a value of 0 corresponds to an income of >50K.
print(model.predict(answers)[0])