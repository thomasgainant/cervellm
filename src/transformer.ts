export class CervellmTransformer {
  vocab: string[];
  vocabSize: number;
  embedSize: number;

  // Paramètres du modèle
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wo: number[][]; // projection vers vocab

  constructor(vocab: string[], embedSize = 8) {
    this.vocab = vocab;
    this.vocabSize = vocab.length;
    this.embedSize = embedSize;

    // Initialisation des matrices aléatoires
    this.Wq = this.randomMatrix(embedSize, embedSize);
    this.Wk = this.randomMatrix(embedSize, embedSize);
    this.Wv = this.randomMatrix(embedSize, embedSize);
    this.Wo = this.randomMatrix(embedSize, this.vocabSize);
  }

  encode(input: string): number[] {
    return input.split('').map(ch => this.vocab.indexOf(ch));
  }

  decode(indices: number[]): string {
    return indices.map(i => this.vocab[i] || '?').join('');
  }

  embed(indices: number[]): number[][] {
    return indices.map(i => {
      const vec = Array(this.embedSize).fill(0);
      vec[i % this.embedSize] = 1; // simple one-hot
      return vec;
    });
  }

  softmax(arr: number[]): number[] {
    const exps = arr.map(x => Math.exp(x));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  dot(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  matMul(a: number[][], b: number[][]): number[][] {
    return a.map(row =>
      b[0].map((_, colIdx) =>
        row.reduce((sum, val, i) => sum + val * b[i][colIdx], 0)
      )
    );
  }

  matVecMul(mat: number[][], vec: number[]): number[] {
    return mat.map(row => this.dot(row, vec));
  }

  weightedSum(weights: number[], vectors: number[][]): number[] {
    const result = Array(this.embedSize).fill(0);
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < this.embedSize; j++) {
        result[j] += weights[i] * vectors[i][j];
      }
    }
    return result;
  }

  attention(Q: number[][], K: number[][], V: number[][]): number[][] {
    const scores = Q.map(q =>
      K.map(k => this.dot(q, k) / Math.sqrt(this.embedSize))
    );
    const weights = scores.map(this.softmax);
    return weights.map((w, i) => this.weightedSum(w, V));
  }

  predict(input: string): number[] {
    const indices = this.encode(input);
    const X = this.embed(indices);
    const Q = this.matMul(X, this.Wq);
    const K = this.matMul(X, this.Wk);
    const V = this.matMul(X, this.Wv);

    const attended = this.attention(Q, K, V);
    const context = attended[attended.length - 1]; // dernier token
    const logits = this.matVecMul(this.Wo, context); // projection vers vocab
    return this.softmax(logits); // proba de chaque token
  }

  loss(pred: number[], targetIndex: number): number {
    return -Math.log(pred[targetIndex] + 1e-9); // cross-entropy
  }

  updateWeights(input: string, targetChar: string, learningRate = 0.1) {
    const targetIdx = this.vocab.indexOf(targetChar);
    const originalLoss = this.loss(this.predict(input), targetIdx);

    const epsilon = 1e-4;

    // Descente de gradient numérique sur Wo
    for (let i = 0; i < this.Wo.length; i++) {
      for (let j = 0; j < this.Wo[i].length; j++) {
        this.Wo[i][j] += epsilon;
        const newLoss = this.loss(this.predict(input), targetIdx);
        const grad = (newLoss - originalLoss) / epsilon;
        this.Wo[i][j] -= epsilon + learningRate * grad;
      }
    }
  }

  randomMatrix(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() * 0.02 - 0.01)
    );
  }
}