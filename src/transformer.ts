import * as fs from "fs";

export class CervellmTransformer {
  vocab: string[];
  vocabSize: number;
  embedSize: number;

  // Paramètres du modèle
  Wq: number[][];
  Wk: number[][];
  Wv: number[][];
  Wo: number[][]; // projection vers vocab

  constructor(vocab: string[], embedSize?:number | null) {
    this.vocab = vocab;
    this.vocabSize = vocab.length;
    this.embedSize = embedSize != null ? embedSize : this.vocabSize;

    // Initialisation des matrices aléatoires
    this.Wq = this.randomMatrix(this.embedSize, this.embedSize);
    this.Wk = this.randomMatrix(this.embedSize, this.embedSize);
    this.Wv = this.randomMatrix(this.embedSize, this.embedSize);
    this.Wo = this.randomMatrix(this.embedSize, this.vocabSize);
  }

  encode(input: string): number[] {
    return input.split('').map(ch => {
      const idx = this.vocab.indexOf(ch);
      if (idx === -1) throw new Error(`Unknown character: "${ch}"`);
      return idx;
    });
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
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }  

  dot(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error(`Dimension mismatch: dot(${a.length}, ${b.length})`);
    }
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
    let loss = -Math.log(pred[targetIndex] + 1e-9);
    if (!isFinite(loss)) throw new Error("Loss exploded");
    return loss; // cross-entropy
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
        //console.log(`Grad[${i}][${j}] = ${grad.toFixed(6)}, Wo = ${this.Wo[i][j].toFixed(6)}`);
        this.Wo[i][j] -= epsilon;
        this.Wo[i][j] -= learningRate * grad;
      }
    }
  }

  randomMatrix(rows: number, cols: number): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() * 0.02 - 0.01)
    );
  }

  saveModel(path: string): void {
    const data = {
      vocab: this.vocab,
      embedSize: this.embedSize,
      Wq: this.Wq,
      Wk: this.Wk,
      Wv: this.Wv,
      Wo: this.Wo
    };
    fs.writeFileSync(path, JSON.stringify(data));
  }
  
  static loadModel(path: string): CervellmTransformer {
    const fs = require("fs");
    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    const model = new CervellmTransformer(data.vocab, data.embedSize);
    model.Wq = data.Wq;
    model.Wk = data.Wk;
    model.Wv = data.Wv;
    model.Wo = data.Wo;
    return model;
  }  
}