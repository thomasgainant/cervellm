import { CervellmTransformer } from './transformer';

const vocab = 'abcdefghijklmnopqrstuvwxyz .,!?'.split('');
const model = new CervellmTransformer(vocab);

//Training data
const trainingData = [
  { input: "hell", target: "o" },
  { input: "ello", target: " " },
  { input: "llo ", target: "w" },
  { input: "lo w", target: "o" },
  { input: "o wo", target: "r" },
  { input: " wor", target: "l" },
  { input: "worl", target: "d" },
];

//Training loop
for (let epoch = 0; epoch < 100; epoch++) {
  let totalLoss = 0;
  for (const sample of trainingData) {
    const pred = model.predict(sample.input);
    const targetIdx = vocab.indexOf(sample.target);
    totalLoss += model.loss(pred, targetIdx);
    model.updateWeights(sample.input, sample.target, 0.1);
  }
  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${totalLoss.toFixed(4)}`);
  }
}

//Try generating
let generated = "hell";
for (let i = 0; i < 10; i++) {
  const context = generated.slice(-4);
  const pred = model.predict(context);
  const nextChar = vocab[pred.indexOf(Math.max(...pred))];
  generated += nextChar;
}
console.log("Generated:", generated);
