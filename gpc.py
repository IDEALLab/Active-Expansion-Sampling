"""
Customized Gaussian processes classification that computes the variance

References:
-----------
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier

Author(s): Wei Chen (wchen459@umd.edu)
"""


import numpy as np
from scipy.linalg import solve, cho_solve, cholesky
from scipy.special import erf

from sklearn.utils.validation import check_is_fitted
from sklearn.base import ClassifierMixin, clone
from sklearn.gaussian_process.gpc import _BinaryGaussianProcessClassifierLaplace

from operator import itemgetter

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]


class _BGPCL(_BinaryGaussianProcessClassifierLaplace):
    
    def fit(self, X, y):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self.rng = check_random_state(self.random_state)

        self.X_train_ = np.copy(X) if self.copy_X_train else X

        # Encode class labels and check that it is a binary classification
        # problem
        label_encoder = LabelEncoder()
        self.y_train_ = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        if self.classes_.size > 2:
            raise ValueError("%s supports only binary classification. "
                             "y contains classes %s"
                             % (self.__class__.__name__, self.classes_))

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [self._constrained_optimization(obj_func,
                                                     self.kernel_.theta,
                                                     self.kernel_.bounds)]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0],
                                                            bounds[:, 1]))
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)

        _, (self.pi_, self.W_sr_, self.L_, _, _) = \
            self._posterior_mode(K, return_temporaries=True)

        return self
                  

class GPClassifier(_BGPCL, ClassifierMixin):
    
    def predict_proba(self, X, get_var=False):
        """Return probability estimates for the test vector X.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        check_is_fitted(self, ["X_train_", "y_train_", "pi_", "W_sr_", "L_"])

        # Based on Algorithm 3.2 of GPML
        K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Line 4
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
        
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
                
        if get_var:
            return f_star, var_f_star
            
        # Line 7:
        # Approximate \int log(z) * N(z | f_star, var_f_star)
        # Approximation is due to Williams & Barber, "Bayesian Classification
        # with Gaussian Processes", Appendix A: Approximate the logistic
        # sigmoid by a linear combination of 5 error functions.
        # For information on how this integral can be computed see
        # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) \
            * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2))) \
            / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()
        
        return np.vstack((1 - pi_star, pi_star)).T

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta)

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Compute log-marginal-likelihood Z and also store some temporaries
        # which can be reused for computing Z's gradient
        Z, (pi, W_sr, L, b, a) = \
            self._posterior_mode(K, return_temporaries=True)

        if not eval_gradient:
            return Z

        # Compute gradient based on Algorithm 5.1 of GPML
        d_Z = np.empty(theta.shape[0])
        # XXX: Get rid of the np.diag() in the next line
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))  # Line 7
        C = solve(L, W_sr[:, np.newaxis] * K)  # Line 8
        # Line 9: (use einsum to compute np.diag(C.T.dot(C))))
        s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) \
            * (pi * (1 - pi) * (1 - 2 * pi))  # third derivative

        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]   # Line 11
            # Line 12: (R.T.ravel().dot(C.ravel()) = np.trace(R.dot(C)))
            s_1 = .5 * a.T.dot(C).dot(a) - .5 * R.T.ravel().dot(C.ravel())

            b = C.dot(self.y_train_ - pi)  # Line 13
            s_3 = b - K.dot(R.dot(b))  # Line 14

            d_Z[j] = s_1 + s_2.T.dot(s_3)  # Line 15

        return Z, d_Z
        
    def _posterior_mode(self, K, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.

        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if self.warm_start and hasattr(self, "f_cached") \
           and self.f_cached.shape == self.y_train_.shape:
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64)

        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            # Line 4
            pi = 1 / (1 + np.exp(-f))
            W = pi * (1 - pi)
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = cholesky(B, lower=True)
            # Line 6
            b = W * f + (self.y_train_ - pi)
            # Line 7
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f = K.dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            lml = -0.5 * a.T.dot(f) \
                - np.log(1 + np.exp(-(self.y_train_ * 2 - 1) * f)).sum() \
                - np.log(np.diag(L)).sum()
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        if return_temporaries:
            return log_marginal_likelihood, (pi, W_sr, L, b, a)
        else:
            return log_marginal_likelihood
            
    def predict_real(self, feature, *args, **kwargs):
        feature = np.array(feature)
        dvalue = self.predict_proba(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
        
    def predict_mean_var(self, feature, K_star=None, *args, **kwargs):
        feature = np.array(feature)
        return self.predict_proba(feature, get_var=True, *args, **kwargs)
            
    def get_mu_nu(self):
        s0 = np.sign(self.y_train_).reshape(-1,1)
        mu = s0.T.dot(self.y_train_ - self.pi_)
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * s0)
        nu = np.einsum("ij,ij->j", v, v)
        return mu[0], nu[0]

        
