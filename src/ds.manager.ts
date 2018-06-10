import * as tf from "@tensorflow/tfjs";

class DsManager {

    numClasses: number;
    xs: tf.Tensor;
    ys: tf.Tensor;

    constructor(numClasses: number) {
      this.numClasses = numClasses;
    }

    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example A tensor representing the example. It can be an image,
     *     an activation, or any other type of Tensor.
     * @param {number} label The label of the example. Should be an umber.
     */
    addExample(example: tf.Tensor, label: number) {
      // One-hot encode the label.
      let y = undefined ;
      if (this.numClasses > 2) {
        y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]), this.numClasses));
      } else {
        y = tf.tensor1d([label]) ;
      }

      if (this.xs) {
        // For the first example that gets added, keep example and y so that the
        // ControllerDataset owns the memory of the inputs. This makes sure that
        // if addExample() is called in a tf.tidy(), these Tensors will not get
        // disposed.
        this.xs = tf.keep(example);
        this.ys = tf.keep(y);
      } else {
        const oldX = this.xs;
        this.xs = tf.keep(oldX.concat(example, 0));

        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));


        oldX.dispose();
        oldY.dispose();
        y.dispose();
      }
    }
  }

  export default DsManager;