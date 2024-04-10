import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

class HMM_Regression:
    def __init__(self, n_components, df, target_col, lag_list, method = "posterior", add_constant = True, difference = None, cat_var = None, drop_categ= None, n_iter = 100, tol=1e-6,coefficients=None, stds = None, init_state = None, trans_matrix= None, eval_set = None):
        self.N = n_components
  
        self.cat_var = cat_var
        self.drop_categ = drop_categ
        self.target_col = target_col
        self.diff = difference
        self.cons = add_constant
        
        self.lag_list = lag_list
        self.df = self.data_prep(df)
        self.X = self.df.drop(columns = self.target_col)
        self.y = self.df[self.target_col]

        if self.cons == True:
            self.X = sm.add_constant(self.X)
        self.col_names = self.X.columns.tolist()
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.T = len(self.y)

        self.Xs = np.array(self.X)
        self.ys = np.array(self.y)

        if init_state is None:
            self.pi = np.full(self.N , 1/self.N )
        else:
            self.pi = init_state
        if trans_matrix is None:
            self.A = np.full((self.N,self.N), 1/self.N)
        else:
            self.A = trans_matrix
            
        if coefficients is None:
            self.coeffs = np.full((self.N, self.X.shape[1]), 1e-06)
        else:
            self.coeffs = coefficients
        
        if stds is None:
            self.stds = np.full((self.N,), np.std(self.y))
        else:
            self.stds = stds

        # self.coeffs = np.full((self.N, self.X.shape[1]), 1e-06)
        self.method = method
        self.iter = n_iter
        self.tol = tol
        self.eval_set = eval_set
        self.optimize()

    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
            
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if self.target_col in dfc.columns:
            if self.diff is not None:
                self.last_train = df[self.target_col].tolist()[-1]
                for i in range(1, self.diff+1):
                    dfc[self.target_col] = dfc[self.target_col].diff(1)
                    
            for i in self.lag_list:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)       
        dfc = dfc.dropna()
        return dfc
        
        
    def compute_forward(self):
        # Forward Algorithm-scale
        self.forward = np.zeros((self.N, self.T))
        self.scales = []
        self.fitted = np.zeros((self.N, self.T))
        for i in range(self.N):  # initialization step
            mu = sum(self.coeffs[i]*self.X[0])
            self.fitted[i, 0] = mu
            pdf_0 = norm.pdf(self.y[0], loc=mu, scale=self.stds[i])
            self.forward[i, 0] = self.pi[i]*pdf_0
            
        c = 1/self.forward[:, 0].sum()
        self.forward[:, 0] = self.forward[:, 0]*c
        self.scales.append(c)
        
        for t in range(1, self.T): # recursion step
            for s in range(self.N):
                mu = sum(self.coeffs[s]*self.X[t])
                self.fitted[s, t] = mu
                pdf_s = norm.pdf(self.y[t], loc=mu, scale=self.stds[s])
                for k in range(self.N):
                    self.forward[s, t] += self.forward[k, t-1]*self.A[k,s]*pdf_s
            c = 1/self.forward[:, t].sum()
            self.forward[:, t] = self.forward[:, t]*c
            self.scales.append(c)
                
        obs_prob = -sum(np.log(self.scales)) # termination step: probability of the observation sequence
        self.LL = obs_prob
        # likelihods.append(obs_prob)
        return obs_prob
    def compute_backward(self):
        # backward algorithm:
        self.backward = np.zeros((self.N, self.T))
        for i in range(self.N): # initialization
            self.backward[i, self.T-1] = 1 #scale probs
        self.backward[:, self.T-1] = self.backward[:, self.T-1]*self.scales[self.T-1]   
            
        for t in range(self.T-2, 0, -1): #recursion
            for s in range(self.N):
                for k in range(self.N):
                    mu = sum(self.coeffs[k]*self.X[t+1])
                    pdf_k = norm.pdf(self.y[t+1], loc=mu, scale=self.stds[k])
                    self.backward[s, t] += self.A[s,k]*pdf_k*self.backward[k,t+1]
            self.backward[:, t] = self.backward[:, t]*self.scales[t] #scale factor
        
        
        
        # for t =0
        for s in range(self.N):
            for k in range(self.N):
                mu = sum(self.coeffs[k]*self.X[1])
                pdf_k = norm.pdf(self.y[1], loc=mu, scale=self.stds[k])
                self.backward[s, 0] += self.A[s,k]*pdf_k*self.backward[k,1]
        
        self.backward[:, 0] = self.backward[:, 0]*self.scales[0] #scale
        
        self.P_ter = 0
        for i in range(self.N):
            mu = sum(self.coeffs[i]*self.X[0])
            pdf_0 = norm.pdf(self.y[0], loc=mu, scale=self.stds[i])
            self.P_ter += self.pi[i]*pdf_0*self.backward[i, 0]
        
        poster = self.backward*self.forward # (at*bt from t to T for all states) # the probality of the observation for whole utterance
        self.total_prob = poster.sum(axis =0)
        self.posterior = poster/self.total_prob
        if self.method =="posterior":
            self.best_path = np.argmax(self.posterior, axis = 0)

    def compute_viterbi(self):
        viterbi = np.zeros((self.N, self.T))
        backpointers = np.zeros((self.N, self.T), dtype=int)
        for i in range(self.N):
            mu = sum(self.coeffs[i]*self.X[0])
            pdf_0 = norm.pdf(self.y[0], loc=mu, scale=self.stds[i])
            viterbi[i, 0] = np.log(self.pi[i]) + np.log(pdf_0)
            backpointers[i, 0] = 0
        # viterbi[:, 0] = viterbi[:, 0]/viterbi[:, 0].sum() #scale probs
            
        for t in range(1, self.T): # recursion step
            for s in range(self.N):
                mu = sum(self.coeffs[s]*self.X[t])
                pdf_s = norm.pdf(self.y[t], loc=mu, scale=self.stds[s])
                paths = [viterbi[k, t-1]+np.log(self.A[k,s])+np.log(pdf_s) for k in range(self.N)]
                viterbi[s, t] = np.max(paths)
                backpointers[s, t] = np.argmax(paths)
            # viterbi[:, t] = viterbi[:, t]/viterbi[:, t].sum()
        bestpathprob = np.max(viterbi[:, self.T-1])
        best_path_point = np.argmax(viterbi[:, self.T-1])
        
        # Backtracking to find the best path
        best_path = [best_path_point]
        for t in range(self.T-1, 0, -1):
            prev_state = backpointers[:, t][best_path[-1]]
            best_path.append(prev_state)
        best_path.reverse()
        if self.method == "viterbi":
            self.best_path = best_path
        self.poster_viterb = np.max(viterbi, axis=0)
    # trainsition prob calculations for each time step:

    def summary(self):
        output = pd.DataFrame()
        for i in range(self.N):
            wi = self.posterior[i]
            W = np.diag(wi)

            X_weight = np.dot(W, self.Xs)
            y_weight= np.dot(W, self.ys)
            coeff_state = np.linalg.lstsq(X_weight, y_weight, rcond=None)[0]
            
            resid = y_weight-(coeff_state*X_weight).sum(axis = 1)
            # statistics:
            RSS = np.sum(resid**2)
            n = len(y_weight)
            p = len(coeff_state)  # Number of coefficients (including intercept)
            SE = np.sqrt(RSS / (n - p) * np.diag(np.linalg.inv(X_weight.T @ X_weight)))

            from scipy.stats import t
            t_values = coeff_state / SE
            # Degrees of freedom
            df = n - p  # degrees of freedom (n - number of coefficients)
            # Calculate p-values for coefficients
            p_values = (1 - t.cdf(np.abs(t_values), df)) * 2

            result = np.round(np.column_stack((coeff_state, SE, t_values, p_values)), 3)
            result = pd.DataFrame(result)
            result.index = [f"sate-{i}_{item}" for item in self.col_names]
            result.columns = ["coefficients", "SE", "t-values", "p-values"]
            output = pd.concat([output, result])
        return output

            
    def EM(self):
        # trainsition prob calculations for each time step:
        fina_eq_all = np.zeros((self.N**2, self.T-1))
        for t in range(self.T-1):
            trans = []
            idx = 0
            for i in range(self.N):
        
                for j in range(self.N):
                    mu = sum(self.coeffs[j]*self.X[t+1])
                    pdf_j = norm.pdf(self.y[t+1], loc=mu, scale=self.stds[j])
                    i_j = self.forward[i, t]*self.A[i, j]*pdf_j*self.backward[j, t+1]
        
                    final_eq = i_j/self.total_prob[t]
                    fina_eq_all[idx, t] = final_eq
                    idx+=1
                    
        # transition prob update          
        new_trans = []
        idx = 0
        for i in range(self.N):
            for j in range(self.N):
                ij = fina_eq_all[idx].sum()/np.array_split(fina_eq_all, self.N, axis=0)[i].sum()
                new_trans.append(ij)
                idx +=1
        self.A = np.array(new_trans).reshape(self.N,self.N)
        
        # compute new coeffs, variance &
        # states = np.sort(pd.Series(self.best_path).drop_duplicates().to_numpy())
        states = np.arange(0,  self.N)
        coeffs = []
        stds = 0
        self.state_vars = []
        # weights = np.argmax(self.posterior, axis=0)
        for i in states:
            # indexes = np.where(np.isin(self.best_path, i))[0]
            wi = self.posterior[i]
            W = np.diag(wi)
        
            # state_y= self.y[indexes]
            # state_x = self.X[indexes]
            # XW = np.dot(self.X.T, W)
            X_weight = np.dot(W, self.X)
            y_weight= np.dot(W, self.y)
            coeff_state = np.linalg.lstsq(X_weight, y_weight, rcond=None)[0]

            coeffs.append(coeff_state)
            # std_state = np.std(y_weight-(coeff_state*X_weight).sum(axis = 1)) #---------STDs
            resid = y_weight-(coeff_state*X_weight).sum(axis = 1)
            d_free = len(self.y)-len(coeff_state)
            self.state_vars.append(np.sqrt(sum(resid**2)/d_free))
            stds += sum(resid**2)/d_free
        
            # std_state = np.std(y_weight-(coeff_state*X_weight).sum(axis = 1)) #---------STDs
            # stds.append(std_state)
        
        self.coeffs = np.row_stack(coeffs)
        # print(self.coeffs)
        self.stds = np.repeat(np.sqrt(stds), self.N)

    def optimize(self):
        self.LLs = []
        # self.diff = []
        for i in range(self.iter):
            ll = self.compute_forward()
            self.compute_backward()
            if self.method == "viterbi":
                self.compute_viterbi()
            self.EM()
            self.predicted = (self.fitted*self.posterior).sum(axis = 0)
            if self.eval_set is not None:
                pred = self.forecast(42, self.eval_set[0])
                # frs_list = list(pred)
                # frs_list.insert(0, self.eval_set[2])
                # preds = np.cumsum(frs_list)[1:]
                mae = mean_absolute_error(self.eval_set[1], np.array(pred))
                # print(mae)
            
            if i>0:
                eps = ll-self.LLs[-1]
                if self.eval_set is not None:
                    print(f"iteration: {i+1}, LL: {ll}, eps: {eps}. Stardard deviations are {self.stds} and mae: {mae}")
                else:
                    print(f"iteration: {i+1}, LL: {ll}, eps: {eps}. Stardard deviations are {self.stds}")
                # self.diff.append(eps)
                if np.abs(eps) < self.tol:
                    print(f"Converged after {i + 1} iterations.")
                    break
            else:
                print(f"iteration: {i+1}, LL: {ll}, eps: {None}")
            self.LLs.append(ll)
            # print("Transition matrix:", self.A)
        

    def forecast(self, H, exog):
        y_list = self.y.tolist()
        if self.cons == True:
            exog = sm.add_constant(exog)
        exog = np.array(self.data_prep(exog))
        
        forecasts = []
        # forecasts2 = []
        f_forward = np.zeros((self.N, self.T))
        state_preds = np.zeros((self.N, H))
        for t in range(H): # recursion step
            exo_inp = exog[t].tolist()
            for s in range(self.N):
                lags = [y_list[-l] for l in self.lag_list]

                inp = exo_inp+lags
                final_inp=np.array(inp)
                mu = sum(self.coeffs[s]*final_inp)
                state_preds[s, t] = mu
                pdf_s = norm.pdf(mu, loc=mu, scale=self.stds[s])
                for k in range(self.N):
                    if t == 0:
                        f_forward[s, t] += self.forward[k, self.T-1]*self.A[k,s]*pdf_s
                    else:
                        f_forward[s, t] += f_forward[k, t-1]*self.A[k,s]*pdf_s
                        
            c = 1/f_forward[:, t].sum()
            f_forward[:, t] = f_forward[:, t]*c
            # max_state_idx = np.argmax(f_forward[:, t])
            # max_state_idx = np.argmin(np.abs(state_preds[:, t]))
            # pred = state_preds[:, t][max_state_idx]
            # forecasts2.append(pred)
            pred_w = sum(f_forward[:, t]*state_preds[:, t])
            forecasts.append(pred_w)
            y_list.append(pred_w)

        if self.diff is not None:
            forecasts.insert(0, self.last_train)
            forecasts = np.cumsum(forecasts)[1:]
        return np.array(forecasts)

    def fit(self, df_train):
        df = self.data_prep(df_train)
        self.X = np.array(df.drop(columns = self.target_col))
        if self.cons == True:
            self.X = sm.add_constant(self.X)
            
        self.y = np.array(df[self.target_col])
        self.T = len(self.y)
        # Forward Algorithm-scale
        self.forward = np.zeros((self.N, self.T))
        self.scales = []
        for i in range(self.N):  # initialization step
            mu = sum(self.coeffs[i]*self.X[0])
            pdf_0 = norm.pdf(self.y[0], loc=mu, scale=self.stds[i])
            self.forward[i, 0] = self.pi[i]*pdf_0
            
        c = 1/self.forward[:, 0].sum()
        self.forward[:, 0] = self.forward[:, 0]*c
        self.scales.append(c)
        
        for t in range(1, self.T): # recursion step
            for s in range(self.N):
                mu = sum(self.coeffs[s]*self.X[t])
                pdf_s = norm.pdf(self.y[t], loc=mu, scale=self.stds[s])
                for k in range(self.N):
                    self.forward[s, t] += self.forward[k, t-1]*self.A[k,s]*pdf_s
            c = 1/self.forward[:, t].sum()
            self.forward[:, t] = self.forward[:, t]*c
            self.scales.append(c)
             
        obs_prob = -sum(np.log(self.scales)) # termination step: probability of the observation sequence
        self.LL = obs_prob
        # likelihods.append(obs_prob)
        return obs_prob