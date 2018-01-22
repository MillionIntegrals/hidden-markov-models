import numpy as np

import hiddenmm.model.markov_chain as mc
import hiddenmm.constants as constants


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

    def likelihood(self, observations: np.ndarray) -> float:
        """ Return likelihood of supplied observation series """
        dimension = observations.shape[0]

        if dimension > 0:
            vector = self.markov_chain.initial * self.projection[:, observations[0]]

            for i in range(1, dimension):
                vector = (vector @ self.transition_matrix) * self.projection[:, observations[i]]

            return vector.sum()
        else:
            return 0.0

    def solve_for_states(self, observations: np.ndarray) -> np.ndarray:
        """ Solve for the most probable state sequence for given observation series """
        dimension = observations.shape[0]

        if dimension > 0:
            result = []

            buffer = np.zeros((dimension, self.num_states), dtype=float)
            state_buffer = np.zeros((dimension, self.num_states), dtype=int)

            buffer[0] = self.initial_distribution * self.projection[:, observations[0]]

            for i in range(1, dimension):
                # Reshape the buffer to make sure the multiplication is broadcasted correctly
                previous_step = buffer[i-1].reshape((self.num_states, 1))
                local_likelihoods = (previous_step * self.transition_matrix)

                new_likelihood = local_likelihoods.max(axis=0)
                state_buffer[i] = local_likelihoods.argmax(axis=0)
                buffer[i] = new_likelihood * self.projection[:, observations[i]]

            final_state = np.argmax(buffer[dimension-1])
            result.append(final_state)

            current_state = final_state

            for i in range(dimension-1, 0, -1):
                if buffer[i][current_state] <= 0.0:
                    raise ValueError("Impossible observation sequence [likelihood = 0].")

                current_state = state_buffer[i][current_state]
                result.append(current_state)

            return np.array(result[::-1], dtype=int)
        else:
            return np.array([], dtype=int)

    def fit_single(self, observations: np.ndarray) -> 'DiscreteHiddenMM':
        """ Perform an expectation-maximization procedure on the hidden markov chain model """
        dimension = observations.shape[0]

        if dimension > 0:
            # Helper variables
            alpha = np.zeros((dimension, self.num_states), dtype=float)
            beta = np.zeros((dimension, self.num_states), dtype=float)

            alpha[0] = self.initial_distribution * self.projection[:, observations[0]]

            for i in range(1, dimension):
                alpha[i] = (alpha[i-1] @ self.transition_matrix) * self.projection[:, observations[i]]

            beta[dimension-1] = 1.0

            for i in range(dimension - 1, 0, -1):
                beta[i-1] = self.transition_matrix @ (self.projection[:, observations[i]] * beta[i])

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
