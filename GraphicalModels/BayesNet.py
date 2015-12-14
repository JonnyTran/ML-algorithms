author = "Nhat Tran"
import random
import sys

class BayesNet:
    def __init__(self, file):
        self.markov = []
        self.parse(file)

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

    def likelihood_monte_carlo(self, confidence=0.95, sample_size=100, n_times=30, query=None):
        if query == None:
            query = self.query

        print "query", query
        the_d = {}

        distribution = self.likelihood_monte_carlo_experiment(sample_size, query)
        for k, v in distribution.items():
            the_d[k] = [v, ]

        for i in range(n_times - 1):
            distribution = self.likelihood_monte_carlo_experiment(sample_size, query)
            for k, v in distribution.items():
                the_d[k].append(v)

        print the_d
        output_file = open('output.txt', 'w')
        for k, v in the_d.items():
            mean, conf_int = self.mean_confidence_interval(v, confidence)
            print "hey", k, mean, conf_int
            output_file.write('{0}, {1}, {2}\n'.format(k, mean, conf_int))

        output_file.write(' , '.join(query) + '\n')
        output_file.close()

    def mean_confidence_interval(self, data, confidence):
        alpha = 1 - confidence
        sum = 0
        for element in data:
            sum += element
        n = len(data)
        mean = sum / n

        var = 0
        for element in data:
            var += (element - mean) ** 2
        var /= n - 1

        std = var ** 0.5
        se = std / n ** 0.5

        h = se * self.z2p(1 - alpha / 2.0)
        return mean, h

    def z2p(self, z):
        """
        from z-score return p-value
        """
        from math import erf, sqrt
        return 0.5 * (1 + erf(z / sqrt(2)))

    def likelihood_monte_carlo_experiment(self, sample_size, query):
        # Find query node
        query_node = None
        for i, val in enumerate(query):
            if val == 'Q':
                query_node = i

        # Find outcome nodes
        evidence_var = {}
        for i, val in enumerate(query):
            if val != 'H' and val != 'Q':
                evidence_var[i] = val

        distribution_accum = {'total': 0.0}
        for i in range(sample_size):
            sampled_tuple, weight = self.sample_network(evidence_var)
            # weight = self.calc_weight(distribution_accum, sampled_tuple, query_node, outcome_values)
            print "tuple", sampled_tuple, weight
            if not distribution_accum.has_key(sampled_tuple[query_node]):
                distribution_accum[sampled_tuple[query_node]] = weight
            else:
                distribution_accum[sampled_tuple[query_node]] += weight

            distribution_accum['total'] += weight

        for k, v in distribution_accum.items():
            if k != 'total':
                distribution_accum[k] /= distribution_accum['total']
        distribution_accum.pop('total')
        print "distribution_accum", distribution_accum

        return distribution_accum

    def sample_network(self, evidence_var):
        prior_values = ['U', 'U', 'U', 'U', 'U']
        for i, val in evidence_var.items():
            prior_values[i] = val

        total_weight = 1

        for node in self.markov:
            prior_values, dist, weight = self.sample_node(node[0], prior_values)
            total_weight *= weight

        return prior_values, total_weight

        # tuple, dist, weight = self.sample_node(2, ['0', '0', '1', 'U', 'U'])
        # print "tuple",tuple, "dist",dist,"weight",weight

    def sample_node(self, node_to_sample, prior_values):
        distribution = []
        prior_nodes = []
        prior_values_dict = {}
        weight = 1

        print
        print "sampling node", node_to_sample
        # if prior_values[node_to_sample] != 'H' and prior_values[node_to_sample] != 'Q':
        #     #return P(e|Parentoutcomes(e))
        #     new_prior = copy.deepcopy(prior_values)
        #     new_prior[node_to_sample] = 'U'
        #     p, d, w = self.sample_node(node_to_sample, new_prior)
        #     for str in d:
        #         if str[0] == prior_values[node_to_sample]:
        #             weight = float(str.split(':')[1])
        #
        # else:

        for i, val in enumerate(prior_values):
            if val != 'U' \
                    and self.markov[node_to_sample][1][0][i] != 'U' \
                    and len(self.markov[node_to_sample][1][0][i]) < 2:
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

        if prior_values[node_to_sample] != 'U' \
                and prior_values[node_to_sample] != 'H' \
                and prior_values[node_to_sample] != 'Q':
            for str in distribution:
                if str[0] == prior_values[node_to_sample]:
                    weight = float(str.split(':')[1])
            print prior_values
            print "node", node_to_sample, "=", prior_values[node_to_sample], "dist", distribution, "weight", weight
        else:
            sampled_value = self.sample_from_distribution(distribution)
            prior_values[node_to_sample] = sampled_value

        return prior_values, distribution, weight

    def sample_from_distribution(self, distribution):
        rand = random.random()
        cumulative_prob = 0

        for str in distribution:
            cumulative_prob += float(str.split(':')[1])
            if rand < cumulative_prob:
                return str.split(':')[0]

    def calc_weight(self, distribution_accum, sampled_tuple, query_node, outcome_values):
        weight = 1.0

        for node, outcome in outcome_values.items():
            print node, outcome
            if sampled_tuple[node] != outcome:
                weight *= 3

        return distribution_accum


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: "python BayesNet.py <input_file>"'
    else:
        bn = BayesNet(sys.argv[1])
        bn.likelihood_monte_carlo(confidence=0.95, sample_size=100, n_times=30, query=None)
