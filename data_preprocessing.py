import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv("data/student_marks_prediction_5000.csv")

print(data.info())
print(data.tail())
print(data.shape)
print(data.isnull().sum())
print(data.duplicated().sum())
print(data.describe())

for i in data.select_dtypes(include="number").columns:
    sns.boxplot(data=data,x=i)
    plt.show() 

sns.scatterplot(data=data, x="Study_Hours", y="Marks")
plt.show()

sns.scatterplot(data=data, x="Attendance", y="Marks")
plt.show()

sns.scatterplot(data=data, x="Previous_Year_CGPA", y="Marks")
plt.show()

sns.scatterplot(data=data, x="Social_Media_Usage", y="Marks")
plt.show()

sns.scatterplot(data=data, x="Courses_Registered", y="Marks")
plt.show()

