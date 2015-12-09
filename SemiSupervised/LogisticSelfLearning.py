import heapq

from Regression.LogisticRegression import LogisticRegressor


class LogisticSelfLearner:
    def __init__(self, learner, dim_no=3, k_order=1, alpha=0.001):
        self.learner = learner
        self.dim_no = dim_no
        self.k_order = k_order
        self.alpha = alpha

    def train(self, labeled_set, unlabeled_set, test, k_samples=5):
        self.labeled_set = labeled_set

        while unlabeled_set:
            learner = self.learner(labeled_set, self.dim_no, self.k_order, self.alpha)
            learner.gradient_descent(gradient_threshold=0.0001)
            print learner.theta
            unlabeled_predictions = learner.test(unlabeled_set, normalize=False)
            top_k = heapq.nlargest(k_samples, unlabeled_predictions)

            for top_k_tuple in top_k:
                labeled_set.append(top_k_tuple[1])
                unlabeled_set.remove(top_k_tuple[1][0])
            print "Accuracy on test:", self.validate(learner, test)
            print

        self.learner = learner

    def validate(self, learner, validation_data):
        X_validation = [x[0] for x in validation_data[:]]
        Y_validation = [x[1] for x in validation_data[:]]
        predictions = learner.test(X_validation, normalize=False)
        Y_predictions = [x[1][1] for x in predictions[:]]

        corrects = 0.0
        for i, predicted in enumerate(Y_predictions):
            if predicted == Y_validation[i]:
                corrects += 1
            else:
                print "misclassified:", X_validation[i]

        return corrects/len(Y_validation)


if __name__ == '__main__':
    W = 1
    M = 0
    train_labeled = [((170, 57, 32), W), ((190, 95, 28), M), ((150, 45, 35), W), ((168, 65, 29), M), ((175, 78, 26), M),
                     ((185, 90, 32), M), ((171, 65, 28), W), ((155, 48, 31), W), ((165, 60, 27), W)]
    train_unlabeled = [(182, 80, 30), (175, 69, 28), (178, 80, 27), (160, 50, 31), (170, 72, 30), (152, 45, 29),
                       (177, 79, 28), (171, 62, 27), (185, 90, 30), (181, 83, 28), (168, 59, 24), (158, 45, 28),
                       (178, 82, 28), (165, 55, 30), (162, 58, 28), (180, 80, 29), (173, 75, 28), (172, 65, 27),
                       (160, 51, 29), (178, 77, 28), (182, 84, 27), (175, 67, 28), (163, 50, 27), (177, 80, 30),
                       (170, 65, 28)]
    test = [((169, 58, 30), W), ((185, 90, 29), M), ((148, 40, 31), W), ((177, 80, 29), M), ((170, 62, 27), W),
            ((172, 72, 30), M), ((175, 68, 27), W), ((178, 80, 29), M)]

    m = LogisticSelfLearner(LogisticRegressor)
    m.train(train_labeled, train_unlabeled, test)
