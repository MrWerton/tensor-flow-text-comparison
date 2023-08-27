import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as tf from '@tensorflow/tfjs-node';

export class AnswerVerification {
    private model: use.UniversalSentenceEncoder | null;

    constructor() {
        this.model = null;
    }

    async loadModel() {
        this.model = await use.load();
    }

    async calculateSimilarity(answerCorrect: string, answerUser: string) {
        if (!this.model) {
            throw Error('The model has not been loaded.');
        }

        const embeddingsAnswerCorrect = await this.model.embed([answerCorrect]);
        const embeddingsAnswerUser = await this.model.embed([answerUser]);

        return await this._calculateCosineSimilarity(embeddingsAnswerCorrect, embeddingsAnswerUser);
    }

    async _calculateCosineSimilarity(embeddingsAnswerCorrect: tf.Tensor, embeddingsAnswerUser: tf.Tensor) {
        const dotProduct = tf.matMul(embeddingsAnswerCorrect, embeddingsAnswerUser.transpose());
        const normAnswerCorrect = tf.norm(embeddingsAnswerCorrect);
        const normAnswerUser = tf.norm(embeddingsAnswerUser);
        const similarity = tf.div(dotProduct, tf.mul(normAnswerCorrect, normAnswerUser));
        return similarity.dataSync()[0];
    }
}
