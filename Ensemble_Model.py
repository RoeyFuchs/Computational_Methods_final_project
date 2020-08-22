from utils import column


class Ensemble_Model:
    # the constructor get a list of  trained models
    def __init__(self, models):
        self.models = models

    def predict(self, x_data):
        y_hat_models = []
        for model in self.models:
            y_hat_models.append((model.predict(x_data)))  # make a predict with every model
        y_hat = []
        for i in range(len(x_data)):  # find the most frequency predict. (can not be equality due to odd number of
            # models)
            lst = column(y_hat_models, i)
            y_hat.append(max(set(lst), key=lst.count))
        return y_hat
