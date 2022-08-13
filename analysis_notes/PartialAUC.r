# Samples on how to run partial AUC calculations
# install.packages("pROC")
library(pROC)
data(aSAH)

# plot partial AUC ROC for 90-100% sensitivity
roc(aSAH$outcome, aSAH$s100b, percent=TRUE, partial.auc=c(100,90), partial.auc.correct=TRUE, partial.auc.focus="sens", plot=TRUE, auc.polygon=TRUE)

# plot partial AUC ROC for 90-100% specificity
roc(aSAH$outcome, aSAH$s100b, percent=TRUE, partial.auc=c(100,90), partial.auc.correct=TRUE, partial.auc.focus="spec", plot=TRUE, auc.polygon=TRUE)

# with CI bootstrapped
roc(aSAH$outcome, aSAH$s100b, percent=TRUE, partial.auc=c(100,90), partial.auc.correct=TRUE, partial.auc.focus="spec", plot=TRUE, auc.polygon=TRUE, ci=TRUE, boot.n=2000)

####
roc(X2021_04_06_for_auc$malignant, X2021_04_06_for_auc$probability, plot=TRUE, print.auc=TRUE)
roc(X2021_04_06_for_auc$malignant, X2021_04_06_for_auc$probability, percent=TRUE, partial.auc=c(100,80), partial.auc.correct=TRUE, partial.auc.focus="sens", plot=TRUE, print.auc=TRUE, auc.polygon=TRUE, ci=TRUE, show.thres=TRUE, print.auc.x=60)

# more details
# https://github.com/xrobin/pROC
# https://web.expasy.org/pROC/screenshots.html