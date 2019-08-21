import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

df=pd.read_csv("data_clean.csv", sep=';')
df = df.drop(['Unnamed: 0'],axis=1)

df.shape #(5011, 48)

#Count of different family sizes
plt.subplots(figsize=(15,8))
sns.countplot(x='Family_Size',hue='Race',data=df,palette="inferno")
plt.xlabel('Family Size')
plt.ylabel('COUNT')
plt.legend(labels=['Mexican Americans','Other Hispanics','Non Hispanic White','Non Hispanic Black',
                   'Non Hispanic Asian','Multi Racial'])
plt.show()


#Variations in Body Mass Index with Age
plt.subplots(figsize=(15,8))
sns.lineplot(x="Age", y="BodyMassIndex",data=df,palette="inferno")
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

#Scatterplot regression curve was plotted to depict the relation between height and weight of the survey participants
plt.subplots(figsize=(15,8))
sns.regplot(x="Height", y="Weight",order=3,data=df,line_kws={"color":"r"})
plt.xlabel('Weight (kg)')
plt.ylabel('Standing Height (cm)')
plt.title('Regression curve for weight vs height of participants')
plt.show()


#Average Blood cell concentration variations in people of different ages
plt.subplots(figsize=(15,8))
sns.lineplot(x="Age", y="Red_blood_cells",data=df,color="r",palette="inferno")
sns.lineplot(x="Age", y="White_blood_cells",data=df,color="b",palette="inferno")
plt.xlabel('Age')
plt.ylabel('concentration')
plt.legend(labels=['Red Blood Cells', 'White Blood Cells'])
plt.show()

#Average Glucose levels in individuals with age
plt.subplots(figsize=(15,8))
sns.lineplot(x="Age", y="Glucose",hue="Gender",data=df,palette="Accent_r",err_style=None)
plt.xlabel('Age')
plt.ylabel('Glucose Levels (mg/dL)')
plt.legend(labels=['Males','Females'])
plt.savefig('glucose.png', dpi=300, bbox_inches='tight')
plt.show()

#Average Insulin levels in individuals with age
plt.subplots(figsize=(15,8))
sns.lineplot(x="Age", y="Insulin",hue="Gender",data=df,palette="Accent_r",err_style=None)
plt.xlabel('Age')
plt.ylabel('Insulin Levels (uU/mL)')
plt.legend(labels=['Males','Females'])
plt.savefig('insulin.png', dpi=300, bbox_inches='tight')
plt.show()

#Average Alcohol Consumption in individuals
plt.subplots(figsize=(15,8))
sns.barplot(x="AlcoholConsumption_yearly", y="AlcoholConsumption_yearly", data=df,
            order=df['AlcoholConsumption_yearly'].value_counts().iloc[:10].index,palette="inferno",
            estimator=lambda x: len(x) / len(df) * 100)
plt.xlabel('Units consumed')
plt.ylabel('Percentage(%) of participants')
