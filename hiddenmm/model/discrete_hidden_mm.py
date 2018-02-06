import numpy as np

import hiddenmm.model.markov_chain as mc
import hiddenmm.constants as constants
import hiddenmm.numeric_util as nutil


class DiscreteHiddenMM:
    """ Class implementing a discrete Hidden Markov Model """

    def __init__(self, markov_chain: mc.MarkovChain, projection: np.ndarray):
        self.markov_chain = markov_chain
        self.projection = projection
        self.num_outputs = self.projection.shape[1]

        if self.markov_chain.num_states != projection.shape[0]:
            raise ValueError("Dimension mismatch: projection dies not match number of states of Markov Chain")

        if not np.all((self.projection.sum(axis=1) - 1.0) < constants.EPSILON):
            raise ValueError("Projection matrix distribution does not sum up to one.")

        if not np.all(self.projection >= 0.0):
            raise ValueError("Projection matrix distribution must be positive.")

    @property
    def num_states(self) -> int:
        """ Number of markov chain states """
        return self.markov_chain.num_states

    @property
    def initial_distribution(self) -> np.ndarray:
        """ Initial state probability distribution """
        return self.markov_chain.initial

    @property
    def transition_matrix(self) -> np.ndarray:
        """ Markov Chain transition matrix """
        return self.markov_chain.transition_matrix

    def generate(self, n: int) -> np.ndarray:
        """ Generate given markov model """
        underlying_chain = self.markov_chain.generate(n)

        result = []

        for i in range(n):
            result.append(np.random.choice(self.num_outputs, p=self.projection[underlying_chain[i]]))

        return np.vstack([underlying_chain, np.array(result, dtype=int)]).T

    def _helper_alpha(self, observations: np.ndarray, stable=False) -> (np.ndarray, np.ndarray):
        """ Return a helper "alpha" variable matrix over time for given observation series """
        alpha = np.zeros((observations.shape[0], self.num_states), dtype=float)
        multipliers = np.ones(observations.shape[0], dtype=float)

        # Initialization
        alpha[0] = self.initial_distribution * self.projection[:, observations[0]]

        if stable:
            alpha[0], multipliers[0] = nutil.rescale_vector(alpha[0])

        # Induction
        for i in range(1, observations.shape[0]):
            alpha[i] = (alpha[i-1] @ self.transition_matrix) * self.projection[:, observations[i]]

            if stable:
                alpha[i], multipliers[i] = nutil.rescale_vector(alpha[i])

        return alpha, multipliers

    def _helper_beta(self, observations: np.ndarray, stable=False) -> (np.ndarray, np.ndarray):
        """ Return a helper "beta" variable matrix over time for given observation series """
        beta = np.zeros((observations.shape[0], self.num_states), dtype=float)
        multipliers = np.ones(observations.shape[0], dtype=float)

        # Initialization
        beta[observations.shape[0]-1] = 1.0
        multipliers[observations.shape[0]-1] = 1.0

        # Induction
        for i in range(observations.shape[0] - 1, 0, -1):
            beta[i-1] = self.transition_matrix @ (self.projection[:, observations[i]] * beta[i])

            if stable:
                beta[i-1], multipliers[i-1] = nutil.rescale_vector(beta[i-1])

        return beta, multipliers

    def _helper_delta_psi(self, observations: np.ndarray, stable=False) -> (np.ndarray, np.ndarray):
        """ Return a helper "delta" variable matrix over time for given observation series """
        delta = np.zeros((observations.shape[0], self.num_states), dtype=float)
        psi = np.zeros((observations.shape[0], self.num_states), dtype=int)

        if stable:
            delta[0] = np.log(self.initial_distribution) + np.log(self.projection[:, observations[0]])
        else:
            delta[0] = self.initial_distribution * self.projection[:, observations[0]]

        for i in range(1, observations.shape[0]):
            # Reshape the buffer to make sure the multiplication is broadcasted correctly
            previous_step = delta[i-1].reshape((self.num_states, 1))

            if stable:
                local_likelihoods = previous_step + np.log(self.transition_matrix)
            else:
                local_likelihoods = previous_step * self.transition_matrix

            new_likelihood = local_likelihoods.max(axis=0)
            psi[i] = local_likelihoods.argmax(axis=0)

            if stable:
                delta[i] = new_likelihood + np.log(self.projection[:, observations[i]])
            else:
                delta[i] = new_likelihood * self.projection[:, observations[i]]

        return delta, psi

    def likelihood(self, observations: np.ndarray) -> float:
        """ Return likelihood of supplied observation series """
        dimension = observations.shape[0]

        # Short-circuit
        if dimension == 0:
            return 0.0

        alpha, multipliers = self._helper_alpha(observations, stable=False)
        return alpha[-1].sum()

    def log_likelihood(self, observations: np.ndarray, stable=True) -> float:
        """ Return likelihood of supplied observation series """
        dimension = observations.shape[0]

        # Short-circuit
        if dimension == 0:
            return 0.0

        alpha, multipliers = self._helper_alpha(observations, stable=stable)
        return np.log(alpha[-1].sum()) - np.log(multipliers).sum()

    def solve_for_states(self, observations: np.ndarray, stable=True) -> np.ndarray:
        """ Solve for the most probable state sequence for given observation series """
        dimension = observations.shape[0]

        # Short-circuit
        if dimension == 0:
            return np.array([], dtype=int)

        delta, psi = self._helper_delta_psi(observations, stable=stable)

        result = []
        final_state = np.argmax(delta[dimension-1])
        result.append(final_state)

        current_state = final_state

        # Walk through the time back and reconstruct the state sequence
        for i in range(dimension-1, 0, -1):

            if stable:
                if delta[i][current_state] <= -np.inf:
                    raise ValueError("Impossible observation sequence [likelihood = 0].")
            else:
                if delta[i][current_state] <= 0.0:
                    raise ValueError("Impossible observation sequence [likelihood = 0].")

            current_state = psi[i][current_state]
            result.append(current_state)

        return np.array(result[::-1], dtype=int)

    def fit_single(self, observations: np.ndarray, stable=True) -> 'DiscreteHiddenMM':
        """ Perform an expectation-maximization procedure on the hidden markov chain model """
        dimension = observations.shape[0]

        if dimension > 0:
            # Helper variables
            alpha, alpha_multipliers = self._helper_alpha(observations, stable=stable)
            beta, beta_multipliers = self._helper_beta(observations, stable=stable)

            # Output will be stored in these variables
            new_initial_distribution = np.zeros(self.num_states, dtype=float)
            new_transition_numerator = np.zeros((self.num_states, self.num_states), dtype=float)
            new_transition_denominator = np.zeros((self.num_states, self.num_states), dtype=float)
            new_projection_numerator = np.zeros((self.num_states, self.num_outputs), dtype=float)
            new_projection_denominator = np.zeros((self.num_states, self.num_outputs), dtype=float)

            # The actual content of the algorithm is iterating the xi matrix through time
            for i in range(dimension):
                gamma = (alpha[i] * beta[i]) / (alpha[i] * beta[i]).sum()

                # This piece is only executed in the first step
                if i == 0:
                    new_initial_distribution = gamma

                # We have to skip the last step as there are one less transitions than observations
                if i < dimension - 1:
                    xi_numerator = (
                            self.transition_matrix *
                            np.outer(alpha[i], self.projection[:, observations[i+1]] * beta[i+1])
                    )
                    xi = xi_numerator / xi_numerator.sum()

                    new_transition_numerator += xi
                    new_transition_denominator += gamma.reshape((self.num_states, 1))

                new_projection_numerator[:, observations[i]] += gamma
                new_projection_denominator += gamma.reshape((self.num_states, 1))

            new_transition = new_transition_numerator / new_transition_denominator
            new_projection = new_projection_numerator / new_projection_denominator

            # A way to handle divisions 0/0
            new_transition = np.where(np.isnan(new_transition), np.eye(self.num_states), new_transition)
            new_projection = np.where(np.isnan(new_projection), 1.0 / self.num_outputs, new_projection)

            return DiscreteHiddenMM(mc.MarkovChain(new_initial_distribution, new_transition), new_projection)
        else:
            return self
