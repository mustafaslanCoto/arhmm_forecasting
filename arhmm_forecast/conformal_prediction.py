import pandas as pd
import numpy as np
class hmm_reg_conformalizer():
    def __init__(self, model, delta, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.model = model
        self.data = model.data
        self.lag = model.lag_list
        self.model_orj = model
        self.model_orj.fit(self.data)
        # self.y_train = model.endog.flatten()
        # self.x_train = model.exog
        self.target_col = model.target_col
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        # self.model_fit = self.model(self.y_train, order= self.order, exog = self.x_train, seasonal_order= self.S_order).fit()
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            train = self.data[:-self.H-i]

            if i !=0:
                test_y = self.data[-self.H-i:-i][self.target_col].values
                if self.data.shape[1]>1:
                    test_x = self.data[-self.H-i:-i].drop(columns = self.target_col)
                else:
                    test_x = None
            else:
                test_y = self.data[-self.H:][self.target_col].values
                if self.data.shape[1]>1:
                    test_x = self.data[-self.H:].drop(columns = self.target_col)
                else:
                    test_x = None

            self.model.fit(train)
            y_pred = self.model.forecast(len(test_y), test_x)

            predictions.append(y_pred)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        y_pred = self.model_orj.forecast(self.H, exog = X)

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs

class hmm_var_conformalizer():
    def __init__(self, model, col_idx, delta, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.model = model
        self.data = model.data
        self.idx = col_idx
        self.model_orj = model
        self.model_orj.fit(self.data)
        # self.y_train = model.endog.flatten()
        # self.x_train = model.exog
        self.target_col = model.target_col
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        # self.model_fit = self.model(self.y_train, order= self.order, exog = self.x_train, seasonal_order= self.S_order).fit()
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            train = self.data[:-self.H-i]

            if i !=0:
                test_y = self.data[-self.H-i:-i][self.target_col[self.idx]].values
                if self.data.shape[1]>2:
                    test_x = self.data[-self.H-i:-i].drop(columns = self.target_col)
                else:
                    test_x = None
            else:
                test_y = self.data[-self.H:][self.target_col[self.idx]].values
                if self.data.shape[1]>2:
                    test_x = self.data[-self.H:].drop(columns = self.target_col)
                else:
                    test_x = None

            self.model.fit(train)
            y_pred = self.model.forecast(len(test_y), test_x)[self.target_col[self.idx]]

            predictions.append(y_pred)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        y_pred = self.model_orj.forecast(self.H, exog = X)[self.target_col[self.idx]]

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs