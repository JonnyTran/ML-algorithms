author = "Nhat Tran"
import random


class BayesNet:
    def __init__(self, file):
        self.markov = []
        self.parse(file)

        for l in self.markov:
            print l

    def parse(self, file):
        self.n_nodes = None

        for line in open(file, 'r'):
            entry = line.replace(' ', '').strip('\n').split(',')
            current_node = None

            # Assign number of nodes
            if self.n_nodes == None:
                self.n_nodes = len(entry)

            # Identify the node number of this entry
            for i, str in enumerate(entry):
                if len(str) > 2 and current_node != i:
                    current_node = i
                    if len(self.markov) == i:
                        self.markov.append([current_node, []])
                    break

            if current_node == None:  # If this line is the query line
                self.query = entry
            else:
                self.markov[current_node][1].append(entry)

    def likelihood_monte_carlo(self, confidence_interval=0.95, sample_size=100, n_times=30, query=None):
        if query == None:
            query = self.query

        for i in range(1):
            print self.sample_network()

    def sample_network(self):
        prior_values = ['U', 'U', 'U', 'U', 'U']
        for node in self.markov:
            prior_values = self.sample_node(node[0], prior_values)
            # print node[0], prior_values

        return prior_values

        # print self.sample_node(1, ['0', 'U', 'U', 'U', 'U'])

    def sample_node(self, node_to_sample, prior_values):
        distribution = []
        prior_nodes = []
        prior_values_dict = {}

        for i, val in enumerate(prior_values):
            if val != 'U' and self.markov[node_to_sample][1][0][i] != 'U':
                prior_nodes.append(i)
                prior_values_dict[i] = val

        # Loop through each entry, to check if it matches the prior_values
        for entry in self.markov[node_to_sample][1]:
            correct_entries_no = 0

            # Check prior values to match previously sampled
            for prior_node in prior_nodes:
                if entry[prior_node] == prior_values_dict[prior_node]:
                    correct_entries_no += 1

            # Check all conditions satisfied
            if correct_entries_no == len(prior_nodes):
                for val in entry:
                    if len(val) > 2:
                        distribution.append(val)

        sampled_value = self.value_from_distribution(distribution)
        prior_values[node_to_sample] = sampled_value

        return prior_values

    def value_from_distribution(self, distribution):
        rand = random.random()
        cumulative_prob = 0

        for str in distribution:
            cumulative_prob += float(str.split(':')[1])
            if rand < cumulative_prob:
                print "HERE", str.split(':')[0], "-", rand, "-", distribution
                return str.split(':')[0]


if __name__ == '__main__':
    bn = BayesNet('input.csv')
    bn.likelihood_monte_carlo(confidence_interval=0.95, sample_size=100, n_times=30, query=None)
