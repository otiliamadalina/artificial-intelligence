import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.close('all')

# pas 2
advertising_df = pd.read_csv('../data-sets/advertising.csv')

# pas 3
print("Advertising Data Head:")
print(advertising_df.head())

print("\nAdvertising Data Info:")
print(advertising_df.info())

print("\nAdvertising Data Describe:")
print(advertising_df.describe())

# curatare date, completare date lipsa cu mediana
for col in ['TV', 'Radio', 'Newspaper', 'Sales']:
    advertising_df[col] = advertising_df[col].fillna(advertising_df[col].median())

# pas 3: grafic jointplot pentru relatia dintre TV si Sales
g1 = sns.jointplot(x='TV', y='Sales', data=advertising_df)
g1.ax_joint.set_xlabel('Buget TV')
g1.ax_joint.set_ylabel('Vanzari (Sales)')
plt.show()

# pas 4: grafic jointplot pentru relatia dintre Radio si Sales
g2 = sns.jointplot(x='Radio', y='Sales', data=advertising_df)
g2.ax_joint.set_xlabel('Buget Radio')
g2.ax_joint.set_ylabel('Vanzari (Sales)')
plt.show()

# pas 5: grafic jointplot hex pentru relatia dintre Newspaper si Sales
g3 = sns.jointplot(x='Newspaper', y='Sales', data=advertising_df, kind='hex')
g3.ax_joint.set_xlabel('Buget Newspaper')
g3.ax_joint.set_ylabel('Vanzari (Sales)')
plt.show()

# pas 6: pairplot pentru vizualizarea relatiilor dintre toate variabilele numerice
sns.pairplot(advertising_df[['TV', 'Radio', 'Newspaper', 'Sales']])
plt.show()

# pas 7: lmplot cu linie de regresie pentru TV vs Sales
sns.lmplot(x='TV', y='Sales', data=advertising_df)
plt.xlabel('Buget TV')
plt.ylabel('Vanzari (Sales)')
plt.show()

# pas 8: pregatire date pentru modelare
# x = variabilele independente (predictori)
# y = variabila dependenta (target)
X = advertising_df[['TV', 'Radio', 'Newspaper']]
y = advertising_df['Sales']

# pas 9: impartire date in set de antrenare si set de testare
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# pas 10: importare model LinearRegression
from sklearn.linear_model import LinearRegression

# pas 11: initializare model de regresie liniara
lm = LinearRegression()

# pas 12: antrenare model pe datele de train
lm.fit(X_train, y_train)

# pas 13: creare DataFrame cu coeficientii modelului pentru fiecare variabila
coef_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print("\nCoeficientii modelului:")
print(coef_df)
print(f"\nIntercept (termen liber): {lm.intercept_}\n")

# pas 14: predictii pe setul de test
predictions = lm.predict(X_test)

# pas 15: grafic valori reale vs valori prezise
plt.scatter(y_test, predictions)
plt.xlabel('Valori reale Sales')
plt.ylabel('Valori prezise Sales')
plt.title('Valori reale vs Valori prezise')
plt.show()

# pas 16: calculare metrici pentru evaluarea performantei modelului
from sklearn import metrics

# MAE (Mean Absolute Error): media erorilor absolute
MAE = metrics.mean_absolute_error(y_test, predictions)
# MSE (Mean Squared Error): media patratelor erorilor
MSE = metrics.mean_squared_error(y_test, predictions)
# RMSE (Root Mean Squared Error): radacina patrata din MSE
RMSE = np.sqrt(MSE)

print(f"MAE: {MAE}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}\n")

# pas 17: analiza reziduurilor (diferente dintre valorile reale si cele prezise)
residuals = y_test - predictions

# Histograma reziduurilor: arata distributia erorilor
sns.histplot(residuals, kde=True)
plt.title('Distributia reziduurilor')
plt.show()

# pas 18: creare DataFrame cu coeficientii modelului pentru fiecare caracteristica
data = {
    'Feature': ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
    'Coefficient': [25.981550, 38.590159, 0.190405, 61.279097]
}
coeficient = pd.DataFrame(data)

print(coeficient)
