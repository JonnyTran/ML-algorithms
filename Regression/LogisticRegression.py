import numpy as np

from LinearRegression import LinearRegressor

class LogisticRegressor(LinearRegressor):

    def __init__(self, training_data, dim_no = 3, k_order = 1, alpha = 0.001):
        LinearRegressor.__init__(self, training_data, dim_no, k_order, alpha, normalize=False)

        # for i, tuple in enumerate(self.X):
        #     print tuple, "-", self.Y[i]

    def h_theta(self, theta, x_features):
        """
        This is the logistic function that uses the sigmoid
        """
        h_theta = LinearRegressor.h_theta(self, theta, x_features)
        return 1.0/(1 + np.power(np.e, -h_theta))

    def gradient(self, theta):
        theta_gradient = [0,] * len(theta)

        for theta_i, val in enumerate(theta):
            h_theta_minus_y_sum = 0

            for X_i, tuple in enumerate(self.X):
                x_poly_features = self.generate_polynomial_features(tuple)
                h_theta_minus_y_sum += (self.Y[X_i] - self.h_theta(theta, x_poly_features)) * x_poly_features[theta_i]

            theta_gradient[theta_i] = self.alpha * (1.0/self.training_size) * h_theta_minus_y_sum

        return theta_gradient

    def predict(self, test_x_tuple):
        h_theta = self.h_theta(self.theta, self.generate_polynomial_features(test_x_tuple))
        predicted_class = np.argmax([1-h_theta, h_theta], axis=0)
        confidence = h_theta**predicted_class * (1-h_theta)**(1-predicted_class)
        # print test_x_tuple, predicted_class, confidence

        return predicted_class, confidence


def main():
    W = 1; M = 0 # Class labels to 1 and 0

    training = [
        ((170, 57, 32), W ),
        ((190, 95, 28), M ),
        ((150, 45, 35), W ),
        ((168, 65, 29), M ),
        ((175, 78, 26), M ),
        ((185, 90, 32), M ),
        ((171, 65, 28), W ),
        ((155, 48, 31), W ),
        ((165, 60, 27), W ),
        ((182, 80, 30), M ),
        ((175, 69, 28), W ),
        ((178, 80, 27), M ),
        ((160, 50, 31), W ),
        ((170, 72, 30), M ),
    ]

    test = [(162, 53, 28), (168, 75, 32), (175, 70, 30), (180, 85, 29)]

    myLogisticRegressor = LogisticRegressor(training, dim_no=3, k_order=1)
    print "Theta:", myLogisticRegressor.gradient_descent(gradient_threshold=0.001)

    print myLogisticRegressor.test(test, normalize=False)



if __name__ == "__main__":
    main()