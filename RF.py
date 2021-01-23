import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pydot

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef


############______________________Data____________________####################

x_train=pd.read_csv('L.txt', sep=' ',header=None)
y_train=pd.read_csv('LL.txt',header=None)

x_test=pd.read_csv('T06.txt', sep=' ',header=None)
y_true=pd.read_csv('TT11.txt',header=None)

# print(x_train.head(10))
# print(y_train.head(10))
# print(x_train.shape)
# print(y_train.shape)
#
# print(x_test.head(10))
# print(y_true.head(10))
# print(x_test.shape)
# print(y_true.shape)

#feature_list=['LULC','dis2cc','dis2sbts','dis2ba','dis2rw','dis2hw','dis2mr','dis2st','dis2wa','dis2fo','neiba','neipo','neimaj','pop','elevation','slope']
class_list=['Developed','undeveloped']


####################_____________Random Forest_____________###############

#n_estimators:number of trees

RF=RandomForestClassifier(n_estimators=100, max_features=8)#n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None
RF.fit(x_train,y_train.values.ravel())

# RF = RandomForestClassifier(max_depth=None)
#
# param_grid = [{'n_estimators':[1, 20, 50,100], #[1, 50, 100]
#                'max_features': [1,8,16]}] #[1, 2, 3]
#
# grid_search = GridSearchCV(RF, param_grid, cv=5, scoring='accuracy', return_train_score=True)
# grid_search.fit(x_train,y_train.values.ravel())
#
# print('The best tuned hyperparameters are:')
# print(grid_search.best_params_)
#
# print('\nAll the hyperparameters for the best Random Forest Model are:')
# print(grid_search.best_estimator_)
#
# print('\nThe Cross Validated Accuracy for the best Random Forest Model is:')
# print(grid_search.best_score_)
#
# print('\nThe Cross Validated Accuracy for all the combinations:')
# print(grid_search.cv_results_)
#
# df= pd.DataFrame(grid_search.cv_results_)
# df.head()


importance=RF.feature_importances_
print(importance)

y_pred=RF.predict(x_test)
y_train_Pred=RF.predict((x_train))

####################___________Confusion Matrix____________################
cnf_matrix_l=pd.DataFrame(confusion_matrix(y_train, y_train_Pred))
print(cnf_matrix_l)
print(accuracy_score(y_train, y_train_Pred))


cnf_matrix=pd.DataFrame(confusion_matrix(y_true, y_pred))
print(cnf_matrix)

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(cohen_kappa_score(y_true, y_pred))
print(matthews_corrcoef(y_true, y_pred))

#####################____________ROC curve___________________#################
# y_pred_probt = RF.predict_proba(x_train)[:,1]
# fprt,tprt, thresholdst=roc_curve(y_train,y_pred_probt)
# print(fprt)
# print(tprt)

y_pred_prob = RF.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
print(fpr)
print(tpr)

# AUCt= roc_auc_score(y_train, y_pred_probt)
# print(AUCt)

AUC= roc_auc_score(y_true, y_pred_prob)
print(AUC)

# create plot
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve_ RF model')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")
plt.show()


# plt.plot(fprt, tprt, label='ROC curve training')
# plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
# _ = plt.xlabel('False Positive Rate')
# _ = plt.ylabel('True Positive Rate')
# _ = plt.title('ROC Curve_ RF model')
# _ = plt.xlim([-0.02, 1])
# _ = plt.ylim([0, 1.02])
# _ = plt.legend(loc="lower right")
#plt.show()
####################_______________Precision-recall curve____________###############

precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

# create plot
plt.plot(precision, recall, label='Precision-recall curve')
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')
_ = plt.legend(loc="lower left")

APS= average_precision_score(y_true, y_pred_prob)
plt.show()

##################____________________Prediction___________________##############

x_predict=pd.read_csv('P16.txt', sep=' ',header=None)
label_pred=RF.predict(x_predict)
label_pred_prob = RF.predict_proba(x_predict)[:,1]

#################___________export probability map______________##################

label_pred_prob=pd.DataFrame(label_pred_prob)
label_pred_prob.to_csv('probabilitymapRF.txt')

#---------------------------Plot-----------------------------------------
# tree = RF.estimators_[10]
# # Export the image to a dot file
# export_graphviz(tree,
#                 out_file = 'forest.dot',
#                 feature_names = feature_list,
#                 class_names=class_list,
#                 rounded = True,
#                 precision = 1,
#                 filled = True)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('forest.dot')
# # Write graph to a png file
# graph.write_png('forest.png')
# graph.write_pdf('forest.pdf')

