import express from 'express';
import { AnswerVerification } from './alg/answer_verification';
const app = express();
app.use(express.json());

app.post('/check-answer', async (request, response) => {
  const { correctAnswer, userAnswer } = request.body;

  try {
    const answerVerification = new AnswerVerification();
    await answerVerification.loadModel();

    const similarity = await answerVerification.calculateSimilarity(correctAnswer, userAnswer);

    response.status(200).json(similarity);

  } catch (error) {

    response.status(500).json(`error: ${error} `)
  }

});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running in http://localhost:${PORT}`);
});

