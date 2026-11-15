"""Small, dependency-free text ML: tokenizer, TF-IDF, softmax regression, Naive Bayes.

Everything here is trained from scratch on project data. No pretrained
embeddings, language models or external APIs are used.
"""
import math
import random
import re

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves using used use well within may two one three new study studies
investigation investigations experiment experiments research data results objective objectives approach nasa""".split())

TOKEN = re.compile(r"[a-z][a-z0-9\-]{1,}")


def tokenize(text, bigrams=True):
    toks = [t.strip("-") for t in TOKEN.findall((text or "").lower())]
    toks = [t for t in toks if t and t not in STOP and len(t) > 1]
    out = list(toks)
    if bigrams:
        out += [a + "_" + b for a, b in zip(toks, toks[1:])]
    return out


class Tfidf:
    def __init__(self, min_df=2, max_features=6000, bigrams=True):
        self.min_df, self.max_features, self.bigrams = min_df, max_features, bigrams
        self.vocab, self.idf = {}, []

    def fit(self, docs):
        df = {}
        for d in docs:
            for t in set(tokenize(d, self.bigrams)):
                df[t] = df.get(t, 0) + 1
        terms = [t for t, c in df.items() if c >= self.min_df]
        terms.sort(key=lambda t: (-df[t], t))
        terms = terms[: self.max_features]
        self.vocab = {t: i for i, t in enumerate(sorted(terms))}
        n = len(docs)
        self.idf = [0.0] * len(self.vocab)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        return self

    def transform_one(self, text):
        tf = {}
        for t in tokenize(text, self.bigrams):
            i = self.vocab.get(t)
            if i is not None:
                tf[i] = tf.get(i, 0) + 1
        vec = {i: (1 + math.log(c)) * self.idf[i] for i, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {i: v / norm for i, v in vec.items()}

    def transform(self, docs):
        return [self.transform_one(d) for d in docs]

    def to_dict(self):
        return {"min_df": self.min_df, "max_features": self.max_features, "bigrams": self.bigrams,
                "vocab": self.vocab, "idf": self.idf}

    @classmethod
    def from_dict(cls, d):
        o = cls(d["min_df"], d["max_features"], d["bigrams"])
        o.vocab, o.idf = d["vocab"], d["idf"]
        return o


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class SoftmaxRegression:
    """Multinomial logistic regression, L2-regularized, full-batch gradient descent
    with class-balanced sample weights."""

    def __init__(self, l2=1e-3, lr=2.0, epochs=300, balanced=True, seed=0):
        self.l2, self.lr, self.epochs, self.balanced, self.seed = l2, lr, epochs, balanced, seed
        self.classes, self.W, self.b = [], [], []

    def fit(self, X, y, n_features):
        self.classes = sorted(set(y))
        K, n = len(self.classes), len(X)
        ci = {c: i for i, c in enumerate(self.classes)}
        yi = [ci[c] for c in y]
        counts = [yi.count(k) for k in range(K)]
        sw = [(n / (K * counts[k])) if self.balanced else 1.0 for k in yi]
        W = [[0.0] * n_features for _ in range(K)]
        b = [0.0] * K
        for ep in range(self.epochs):
            gW = [dict() for _ in range(K)]
            gb = [0.0] * K
            for x, k_true, w in zip(X, yi, sw):
                z = [b[k] + sum(W[k][j] * v for j, v in x.items()) for k in range(K)]
                p = softmax(z)
                for k in range(K):
                    g = (p[k] - (1.0 if k == k_true else 0.0)) * w
                    gb[k] += g
                    gk = gW[k]
                    for j, v in x.items():
                        gk[j] = gk.get(j, 0.0) + g * v
            lr = self.lr / (1 + ep * 0.01)
            for k in range(K):
                Wk = W[k]
                if self.l2:
                    decay = 1 - lr * self.l2
                    for j in range(n_features):
                        Wk[j] *= decay
                for j, g in gW[k].items():
                    Wk[j] -= lr * g / n
                b[k] -= lr * gb[k] / n
        self.W, self.b = W, b
        return self

    def predict_proba_one(self, x):
        z = [self.b[k] + sum(self.W[k][j] * v for j, v in x.items()) for k in range(len(self.classes))]
        return softmax(z)

    def predict_proba(self, X):
        return [self.predict_proba_one(x) for x in X]

    def to_dict(self):
        # store sparse weights to keep the artifact small
        sparse = [{str(j): round(w, 6) for j, w in enumerate(row) if abs(w) > 1e-6} for row in self.W]
        return {"type": "softmax_regression", "classes": self.classes, "b": self.b, "W_sparse": sparse,
                "hyperparameters": {"l2": self.l2, "lr": self.lr, "epochs": self.epochs, "balanced": self.balanced}}

    @classmethod
    def from_dict(cls, d, n_features):
        hp = d["hyperparameters"]
        o = cls(hp["l2"], hp["lr"], hp["epochs"], hp["balanced"])
        o.classes, o.b = d["classes"], d["b"]
        o.W = []
        for row in d["W_sparse"]:
            r = [0.0] * n_features
            for j, w in row.items():
                r[int(j)] = w
            o.W.append(r)
        return o


class MultinomialNB:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y, n_features):
        self.classes = sorted(set(y))
        K = len(self.classes)
        self.prior = [math.log(y.count(c) / len(y)) for c in self.classes]
        self.loglik = []
        for c in self.classes:
            tot = [self.alpha] * n_features
            for x, yy in zip(X, y):
                if yy == c:
                    for j, v in x.items():
                        tot[j] += v
            s = sum(tot)
            self.loglik.append([math.log(t / s) for t in tot])
        return self

    def predict_proba_one(self, x):
        z = [self.prior[k] + sum(self.loglik[k][j] * v for j, v in x.items()) for k in range(len(self.classes))]
        return softmax(z)

    def predict_proba(self, X):
        return [self.predict_proba_one(x) for x in X]


class Majority:
    def fit(self, X, y, n_features):
        self.classes = sorted(set(y))
        top = max(self.classes, key=lambda c: y.count(c))
        self.p = [1.0 if c == top else 0.0 for c in self.classes]
        return self

    def predict_proba(self, X):
        return [list(self.p) for _ in X]


def group_kfold(groups, k, seed):
    """Assign whole groups to folds (greedy balancing after a seeded shuffle)."""
    uniq = sorted(set(groups))
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    sizes = {g: groups.count(g) for g in uniq}
    uniq.sort(key=lambda g: -sizes[g])  # stable after shuffle for ties
    load = [0] * k
    fold_of = {}
    for g in uniq:
        f = min(range(k), key=lambda i: (load[i], i))
        fold_of[g] = f
        load[f] += sizes[g]
    return [fold_of[g] for g in groups]
