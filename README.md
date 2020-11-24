
dataset=pd.read_csv("D:\Python37\salary")
dataset.head(10)
plt.scatter(dataset['Years of experience'],dataset['salary'])
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
X=dataset.iloc[:,:-1]             #in array format 
Y=dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split 
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=1/3,random_state=101)
#Linear Regression
from sklearn.linear_model import LinearRegression
LR=LinearRegression
LR.fit(X_train,Y_train) //model got trained
ypred_LR=LR.predict(X_test)
diff_LR=y_test-ypred_LR
res_df=pd.concat([pd.Series(y_pred_LR),pd.series(y_test),pd.series(diff_LR)],axis=1
res.df.coloumns=['Prediction',OriginalDta','Difference']
plt.scatter(X_train,Y_train,color='blue')
plt.scatter(X_train,LR.predict(Y_train),color='red')
#now check for test data
plt.scatter(X_train,Y_train,color='blue')
plt.scatter(X_train,LR.predict(Y_train),color='red')
#metrics//rmse mse r2
from sklearn import metrics
rmse=numpy.sqrt(mean.squared_error(y_test,y_pred_LR))
R2=metrics.r2_score(y_test,y_pred_LR)# IF r2 is more than 70% we can deploy we gwt r2 as 97% so we can use this model
rmse 
r2
LR.predict([[6]])//83000 compare with actual data actual is 93000



h


