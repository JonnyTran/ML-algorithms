import matplotlib.pyplot as plt
import numpy as np


class LinearRegressor:

    def __init__(self, training_data, dim_no = 3, k_order = 1, alpha = 0.02, normalize=False):
        self.k_order = k_order
        self.dim_no = dim_no
        self.theta = (0,) * (dim_no) * (k_order) + (0,)
        self.alpha = alpha

        self.X = []
        self.Y = []
        for tuple in training_data:
            (self.X).append(tuple[0])
            (self.Y).append(tuple[1])
        self.X = np.array(self.X)
        if normalize:
            self.X = self.normalize(self.X)
        self.Y = np.array(self.Y)
        self.training_size = len(training_data)

    @staticmethod
    def normalize(X):
        X /= np.max(np.abs(X),axis=0)
        return X

    def load_training_data(self, training_data):
        for tuple in training_data:
            (self.X).append(tuple[0])
            (self.Y).append(tuple[1])

    def load_test_data(self, test_data):
        self.test_data = test_data

    def mse_cost(self, theta):
        sum_squared_error = 0

        for X_i, X_tuple in enumerate(self.X):
            sum_squared_error += np.power(self.h_theta(theta, self.generate_polynomial_features(X_tuple)) - self.Y[X_i], 2)

        return sum_squared_error / (2.0*self.training_size)

    @staticmethod
    def mse(Y_truth, Y_predicted):
        sum_squared_error = 0
        print Y_truth, Y_predicted

        for i, y in enumerate(Y_truth):
            sum_squared_error += np.power(y - Y_predicted[i], 2)

        return sum_squared_error / (2.0 * len(Y_truth))

    def h_theta(self, theta, x_features):
        """
        This is the h_theta(x) function
        """
        sum = self.theta[0]
        for k in range(0, self.k_order):
            for d in range(0, self.dim_no):
                sum += theta[k*self.dim_no+d+1] * x_features[k*self.dim_no+d+1]

        return sum

    def generate_polynomial_features(self, x_tuple):
        features = [1,] * len(self.theta)

        for k in range(0, self.k_order):
            for d in range(0, self.dim_no):
                features[k*self.dim_no+d+1] = np.power(x_tuple[d], k+1)

        return features

    def gradient(self, theta):
        theta_gradient = [0,] * len(theta)

        for theta_i, val in enumerate(theta):
            h_theta_minus_y_sum = 0

            for X_i, tuple in enumerate(self.X):
                x_poly_features = self.generate_polynomial_features(tuple)
                h_theta_minus_y_sum += (self.h_theta(theta, x_poly_features) - self.Y[X_i]) * x_poly_features[theta_i]

            theta_gradient[theta_i] = -self.alpha * (1.0/self.training_size) * h_theta_minus_y_sum

        return theta_gradient

    def gradient_descent(self, gradient_threshold = 0.01):
        iteration = 1
        gradient = self.gradient(self.theta)
        new_theta = np.add(self.theta, gradient)
        self.theta = (new_theta)

        while (np.sum(np.absolute(gradient)) >= gradient_threshold or iteration < 500): # Repeat until gradient is small
            # print "Error:",self.mse_cost(new_theta), ", gradient:", np.sum(np.absolute(gradient))
            gradient = self.gradient(self.theta)
            new_theta = np.add(self.theta, gradient)
            self.theta = (new_theta)
            iteration += 1

        print "iteration:", iteration
        return self.theta

    def get_theta(self):
        return self.theta

    def predict(self, test_tuple):
        return self.h_theta(self.theta, self.generate_polynomial_features(test_tuple))

    def test(self, test_data, normalize=False):
        predictions = []
        if normalize:
            test_data = LinearRegressor.normalize(test_data)
        for tuple in test_data:
            predicted_class, confidence = self.predict(tuple)
            predictions.append((confidence, (tuple, predicted_class)))

        return predictions

    def validate_mse(self, validation_data, normalize=False):

        X_validation = [x[0] for x in validation_data[:]]
        if normalize:
            X_validation = self.normalize(X_validation)
        Y_validation = [x[1] for x in validation_data[:]]

        predictions = self.test(X_validation)
        Y_predictions = [x[1] for x in predictions[:]]

        return LinearRegressor.mse(Y_validation, Y_predictions)



def main():
    training = [
        ((6.4432, 9.6309), 50.9155),((3.7861, 5.4681), 29.9852),((8.1158, 5.2114), 42.9626),((5.3283, 2.3159), 24.7445),
        ((3.5073, 4.8890), 27.3704),((9.3900, 6.2406), 51.1350),((8.7594, 6.7914), 50.5774),((5.5016, 3.9552), 30.5206),
        ((6.2248, 3.6744), 31.7380),((5.8704, 9.8798), 49.6374),((2.0774, 0.3774), 10.0634),((3.0125, 8.8517), 38.0517),
        ((4.7092, 9.1329), 43.5320),((2.3049, 7.9618), 33.2198),((8.4431, 0.9871), 31.1220),((1.9476, 2.6187), 16.2934),
        ((2.2592, 3.3536), 19.3899),((1.7071, 6.7973), 28.4807),((2.2766, 1.3655), 13.6945),((4.3570, 7.2123), 36.9220),
        ((3.1110, 1.0676), 14.9160),((9.2338, 6.5376), 51.2371),((4.3021, 4.9417), 29.8112),((1.8482, 7.7905), 32.0336),
        ((9.0488, 7.1504), 52.5188),((9.7975, 9.0372), 61.6658),((4.3887, 8.9092), 42.2733),((1.1112, 3.3416), 16.5052),
        ((2.5806, 6.9875), 31.3369),((4.0872, 1.9781), 19.9475),((5.9490, 0.3054), 20.4239),((2.6221, 7.4407), 32.6062),
        ((6.0284, 5.0002), 35.1676),((7.1122, 4.7992), 38.2211),((2.2175, 9.0472), 36.4109),((1.1742, 6.0987), 25.0108),
        ((2.9668, 6.1767), 29.8861),((3.1878, 8.5944), 37.9213),((4.2417, 8.0549), 38.8327),((5.0786, 5.7672), 34.4707)
    ]

    validation_data = [
        ((0.8552, 1.8292), 11.5848),((2.6248, 2.3993), 17.6138),((8.0101, 8.8651), 54.1331),((0.2922, 0.2867), 5.7326),
        ((9.2885, 4.8990), 46.3750),((7.3033, 1.6793), 29.4356),((4.8861, 9.7868), 46.4227),((5.7853, 7.1269), 40.7433),
        ((2.3728, 5.0047), 24.6220),((4.5885, 4.7109), 29.7602)
    ]

    polynomial_to_error = []

    for k_order_polynomial in [1,2,3,4]:
        print "Linear Regression Gradient Descend on", k_order_polynomial, "order polynomial"

        myLinearRegressor = LinearRegressor(training, dim_no=2, k_order=k_order_polynomial)
        myLinearRegressor.gradient_descent()

        print "Theta:", myLinearRegressor.get_theta()

        # Validate data
        polynomial_to_error.append([k_order_polynomial, myLinearRegressor.validate_mse(validation_data)])

        ############################################# Plot Data Points #####################################################
        fig = plt.figure('%d order polynomial' % (k_order_polynomial,))
        ax = fig.add_subplot(111, projection='3d')

        X1 = []; X2 = []; Y = [];
        for x, y in zip(myLinearRegressor.X, myLinearRegressor.Y):
            X1.append(x[0])
            X2.append(x[1])
            Y.append(y)

        ax.scatter(X1, X2, Y, c='r', marker='o')

        hyp_surf_X1 = np.arange(np.min(X1), np.max(X1), 0.1)
        hyp_surf_X2 = np.arange(np.min(X2), np.max(X2), 0.1)
        hyp_surf_X1, hyp_surf_X2 = np.meshgrid(hyp_surf_X1, hyp_surf_X2)
        hyp_surf_Y = [myLinearRegressor.predict((i,j)) for i,j in zip(hyp_surf_X1, hyp_surf_X2)]

        surf = ax.plot_surface(hyp_surf_X1, hyp_surf_X2, hyp_surf_Y,
                               rstride=1, cstride=1, alpha=0.5,
                                linewidth=0, antialiased=False)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        plt.show()
    ####################################################################################################################

    print "polynomial_to_error", polynomial_to_error

    fig.clear()
    fig = plt.figure("MSE for each k order polynomial")
    plt.plot([tuple[0] for tuple in polynomial_to_error], [tuple[1] for tuple in polynomial_to_error], 'ro')
    plt.axis([0, 5, 0, 7])
    plt.xlabel('K order polynomial hypothesis')
    plt.ylabel('Mean Squared Error')
    plt.show()


if __name__ == "__main__":
    main()