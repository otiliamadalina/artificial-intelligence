import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('Loan_data.csv')

print(train.head())
print(train.info())
print(train.isnull().sum())

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# countplot() - pentru caategorii
# Distributia clasei Loan_Status
sns.set_style('whitegrid')
sns.countplot(x='Loan_Status', data=train, palette='RdBu_r')
plt.show()

# Distributia Loan_Status in functie de gen
sns.set_style('whitegrid')
sns.countplot(x='Loan_Status', hue='Gender', data=train, palette='RdBu_r')
plt.show()

# Distributia Loan_Status in functie de casatorie
sns.set_style('whitegrid')
sns.countplot(x='Loan_Status', hue='Married', data=train, palette='RdBu_r')
plt.show()

# Distributia Loan_Status in functie de istoricul creditului
sns.set_style('whitegrid')
sns.countplot(x='Loan_Status', hue='Credit_History', data=train, palette='RdBu_r')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Loan_Status', hue='Education', data=train, palette='RdBu_r')
plt.show()




# hist() - pentru date numerice
# Histogram pentru ApplicantIncome
# Distributia veniturilor aplicantilor
plt.figure(figsize=(8,5))
plt.hist(train['ApplicantIncome'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distributia ApplicantIncome')
plt.xlabel('ApplicantIncome')
plt.ylabel('Frecventa')
plt.show()

# Histogram pentru CoapplicantIncome
# Distributia veniturilor co-aplicantilor
plt.figure(figsize=(8,5))
plt.hist(train['CoapplicantIncome'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Distributia CoapplicantIncome')
plt.xlabel('CoapplicantIncome')
plt.ylabel('Frecventa')
plt.show()

# Histogram pentru LoanAmount
# Distributia sumelor imprumutului
plt.figure(figsize=(8,5))
plt.hist(train['LoanAmount'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Distributia LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frecventa')
plt.show()



# Curatirea datelor lipsa
# Modificare datelor categorice in numerice pentru a putea face boxplot-uri
train['Loan_Status'] = train['Loan_Status'].map({'N': 0, 'Y': 1})

plt.figure(figsize=(12, 7))
sns.boxplot(x='Loan_Status', y='LoanAmount', data=train, palette='winter')
plt.show()
print(train.groupby('Loan_Status')['LoanAmount'].median())

plt.figure(figsize=(12, 7))
sns.boxplot(x='Loan_Status', y='Loan_Amount_Term', data=train, palette='winter')
plt.show()
print(train.groupby('Loan_Status')['Loan_Amount_Term'].median())


print(train.groupby('Loan_Status')['Credit_History'].median())
print(train.groupby('Self_Employed')['ApplicantIncome'].median())
print(train.groupby('Gender')['ApplicantIncome'].median())
print(train.groupby('Married')['ApplicantIncome'].median())
print(train.groupby('Married')['ApplicantIncome'].median())

def completeaza_loan_amount(row):
    if pd.notnull(row['LoanAmount']):
        return row['LoanAmount']

    if row['Loan_Status'] == 1:  # modificat din '1' in 1
        return 126.0 #126 este o val. aproximata de pe boxplot
    elif row['Loan_Status'] == 0:  # modificat din '0' in 0
        return 129.0
    else:
        return 127.5

train['LoanAmount'] = train.apply(completeaza_loan_amount, axis=1)

def completeaza_loan_term(row):
    if pd.notnull(row['Loan_Amount_Term']):
        return row['Loan_Amount_Term']

    if row['Loan_Status'] in [1, 0]:  # modificat din ['1', '0'] in [1, 0]
        return 360.0 # date luate din graficul boxplot (360 cel mai frecvent)
    else:
        return 360.0

train['Loan_Amount_Term'] = train.apply(completeaza_loan_term, axis=1)

def completeaza_credit_history(row):
    if pd.notnull(row['Credit_History']):
        return row['Credit_History']
    # Logica bancara: creditul aprobat -> istoric bun, creditul respins -> istoric slab
    # Confirmat de groupby('Loan_Status')['Credit_History'].median() -> 0: 1.0, 1: 1.0
    if row['Loan_Status'] == 1:
        return 1  # aprobat -> istoric bun
    elif row['Loan_Status'] == 0:
        return 0  # respins -> istoric slab
    else:
        return 1  # fallback -> istoric bun (valoarea mai frecventa)

train['Credit_History'] = train.apply(completeaza_credit_history, axis=1)


def completeaza_self_employed(row):
    if pd.notnull(row['Self_Employed']):
        return row['Self_Employed']
    # Pragurile luate din groupby('Self_Employed')['ApplicantIncome'].median()
    # No: 3705 / Yes: 5809
    if row['ApplicantIncome'] > 5809:
        return 'Yes'  # venit peste mediana independentilor -> probabil independent
    elif row['ApplicantIncome'] < 3705:
        return 'No'   # venit sub mediana angajatilor -> probabil angajat
    else:
        return train['Self_Employed'].mode()[0]  # venit intre praguri -> mode()

train['Self_Employed'] = train.apply(completeaza_self_employed, axis=1)


def completeaza_gender(row):
    if pd.notnull(row['Gender']):
        return row['Gender']
    # groupby('Gender')['ApplicantIncome'].median() -> Female: 3583, Male: 3865
    # Diferenta prea mica pentru a fi folosita ca predictor -> completam cu mode()
    return train['Gender'].mode()[0]

train['Gender'] = train.apply(completeaza_gender, axis=1)


def completeaza_married(row):
    if pd.notnull(row['Married']):
        return row['Married']
    # Daca are dependenti -> aproape sigur casatorit
    if row['Dependents'] != '0' and pd.notnull(row['Dependents']):
        return 'Yes'
    # groupby('Married')['ApplicantIncome'].median() -> No: 3750, Yes: 3854
    # Diferenta prea mica -> completam cu mode()
    return train['Married'].mode()[0]

train['Married'] = train.apply(completeaza_married, axis=1)


def completeaza_dependents(row):
    if pd.notnull(row['Dependents']):
        return row['Dependents']
    if row['Married'] == 'Yes':
        # Pragurile reutilizate din groupby('Self_Employed')['ApplicantIncome'].median()
        # Venit mare -> mai multi dependenti, venit mic -> mai putini
        if row['ApplicantIncome'] > 5809:
            return '2'  # venit mare -> 2 dependenti
        elif row['ApplicantIncome'] > 3705:
            return '1'  # venit mediu -> 1 dependent
        else:
            return '0'  # venit mic -> 0 dependenti
    else:
        return '0'  # necasatorit -> 0 dependenti

train['Dependents'] = train.apply(completeaza_dependents, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
print(train.isnull().sum())

train = train.drop('Loan_ID', axis=1)

# Coloane categorice in binare
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
train['Married'] = train['Married'].map({'Yes': 1, 'No': 0})
train['Education'] = train['Education'].map({'Graduate': 1, 'Not Graduate': 0})
train['Self_Employed'] = train['Self_Employed'].map({'Yes': 1, 'No': 0})


# Dependents are valori '0','1','2','3+'
def convert_dependents(val):
    if val == '3+':
        return 3
    else:
        return int(val)

train['Dependents'] = train['Dependents'].apply(convert_dependents)

train = pd.get_dummies(train, columns=['Property_Area'], drop_first=True)

print(train.info())

# Impartirea datelor in set de antrenare si testare
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train.drop('Loan_Status', axis=1),
    train['Loan_Status'],
    test_size=0.30,
    random_state=101
)

# Crearea modelului de regresie logistica
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000, class_weight='balanced')
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

# Evaluarea modelului antrenat
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NU', 'DA'],
            yticklabels=['NU', 'DA'])

plt.xlabel('Prezis')
plt.ylabel('Real')
plt.show()

print(classification_report(y_test, predictions))