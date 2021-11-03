import numpy as np


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


# Filter by frequency
def filter_by_freq(T: np.array, freqs: list[int], max_freq_sum: int = 20) -> tuple[int, list]:
    new_T = []
    for T_d in T:
        T_d = sorted(T_d, key=lambda term: freqs[term[0]], reverse=True)
        new_T_d = []
        freq_sum = 0
        for term in T_d:
            freq_sum += term[1]
            if freq_sum > max_freq_sum:
                break
            new_T_d.append(term)
        new_T.append(new_T_d)
    return len(new_T), new_T


if __name__ == '__main__':
    # Configurations
    K = 10
    threshold = 0.01
    top_k = 10

    # Read data and filter
    W, labels, freqs = read_vocab_file('20news/20news.vocab')
    _, T = read_libsvm_file('20news/20news.libsvm')
    D, T = filter_by_freq(T, freqs)

    # Initializations
    np.random.seed(19981011)
    mu = np.random.randn(K, W)
    mu = mu.T / np.sum(mu, axis=1)
    pi = np.random.randn(K)
    pi = pi / np.sum(pi)

    # Pre-process
    n_terms = 0
    fnd_div_ftdw = np.zeros(shape=(D,))
    for d in range(D):
        nd = sum([item[1] for item in T[d]])
        f = np.math.factorial(nd)
        for (w, freq) in T[d]:
            n_terms += freq
            f /= np.math.factorial(freq)
        fnd_div_ftdw[d] = f

    # EM algorithm
    n_rounds = 0
    while True:
        # Obtain $P_{dk}$
        p = np.zeros(shape=(D, K))
        for d in range(D):
            for k in range(K):
                prod_mu = fnd_div_ftdw[d]
                for (w, freq) in T[d]:
                    prod_mu *= np.math.pow(mu[w, k], freq)
                p[d, k] = prod_mu

        # Obtain $\gamma(z_{dk})$
        gamma = p * pi  # [D, K]
        gamma = (gamma.T / np.sum(gamma, axis=1)).T

        # Calculate the new $\pi$
        pi = np.sum(gamma.T, axis=1) / D

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
