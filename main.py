import argparse
import numpy as np
import scipy.special as ss


# Return labels and frequency of terms
def read_vocab_file(path: str) -> tuple[int, list[str], list[int]]:
    with open(path, 'r') as file:
        labels, freqs = [], []
        for line in file.readlines():
            _, label, freq = line.split()
            labels.append(label)
            freqs.append(int(freq))
        return len(labels), labels, freqs


# Return $T_{dw}$ in the formulas
def read_libsvm_file(path: str) -> tuple[int, list]:
    with open(path, 'r') as file:
        T = []
        for line in file.readlines():
            terms = line.split()[1:]
            T_d = []
            for term in terms:
                w, freq = term.split(':')
                w, freq = int(w), int(freq)
                T_d.append((w, freq))
            T.append(T_d)
        return len(T), T


def log_factorial(n: int) -> float:
    return sum([np.math.log(i + 1.0) for i in range(n)])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='EM Algorithm for 20News')
    parser.add_argument('--k', default=20, type=int, help='Number of clusters')
    args = parser.parse_args()

    # Configurations
    K = args.k
    threshold = 0.01
    top_k = 10

    # Read data and filter
    W, labels, freqs = read_vocab_file('20news/20news.vocab')
    D, T = read_libsvm_file('20news/20news.libsvm')

    # Initializations
    np.random.seed(19981011)
    mu = np.random.uniform(1.0, 10.0, (K, W))
    mu = mu.T / np.sum(mu, axis=1)
    pi = np.random.uniform(1.0, 10.0, (K,))
    pi = pi / np.sum(pi)
    # mu = np.ones(shape=(W, K)) / W
    # pi = np.ones(shape=(K, )) / K

    # Pre-process
    n_terms = 0
    log_fnd_div_ftdw = np.zeros(shape=(D,))
    for d in range(D):
        nd = sum([item[1] for item in T[d]])
        f = log_factorial(nd)
        for (w, freq) in T[d]:
            n_terms += freq
            f -= log_factorial(freq)
        log_fnd_div_ftdw[d] = f

    # EM algorithm
    n_rounds = 0
    while True:
        # Obtain $P_{dk}$
        log_p = np.zeros(shape=(D, K))
        for d in range(D):
            for k in range(K):
                prod_mu = log_fnd_div_ftdw[d]
                for (w, freq) in T[d]:
                    prod_mu += freq * np.math.log(mu[w, k])
                log_p[d, k] = prod_mu

        # Obtain $\gamma(z_{dk})$
        log_gamma_num = log_p + np.log(pi)  # [D, K]
        log_gamma_deno = ss.logsumexp(log_gamma_num, axis=1)  # [D]
        log_gamma = (log_gamma_num.T / log_gamma_deno).T  # [D, K]

        # Calculate the new $\pi$
        log_pi = ss.logsumexp(log_gamma.T, axis=1)
        log_pi -= np.math.log(D)

        # Restore from log-scale
        gamma = np.exp(log_gamma)
        pi = np.exp(log_pi)

        # Calculate $\mu$ of the next round
        next_mu = np.zeros_like(mu)
        for d in range(D):
            for (w, freq) in T[d]:
                for k in range(K):
                    next_mu[w, k] += gamma[d, k] * freq
        next_mu /= n_terms
        next_mu /= pi

        # Update
        delta = np.sum(np.abs(mu - next_mu))
        n_rounds += 1
        print('Delta after {} round(s): {}'.format(n_rounds, delta))
        mu = next_mu
        if delta < threshold:
            break

    # Prediction (clustering)
    cluster_count = np.zeros(shape=(K,), dtype=int)
    cluster_freq = np.zeros(shape=(K, W), dtype=int)
    choices = np.argmax(gamma, axis=1)
    for d in range(D):
        cluster_count[choices[d]] += 1
        for (w, freq) in T[d]:
            cluster_freq[choices[d], w] += freq
    print('Results:')
    for k in range(K):
        print(' > {:.0f} items in cluster#{:.0f}'.format(cluster_count[k], k))
        sort_w = np.argsort(-cluster_freq, axis=1)
        for i in range(min(top_k, sort_w.shape[1])):
            if cluster_freq[k, sort_w[k, i]] == 0:
                break
            print('  > Top {} word: {}'.format(i + 1, labels[sort_w[k, i]]))
