import "@tensorflow/tfjs-node";
import * as tf from "@tensorflow/tfjs";
import dsManager from "./ds.manager";
import DsManager from "./ds.manager";
import fs from "fs";


tf.setBackend("tensorflow");


const readImages = (): DsManager => {
    const dsManager = new DsManager(2);


    const img = tf.tidy(() => {
        // Reads the image as a Tensor from the webcam <video> element.
        const image = fs.readFileSync("");
        const webcamImage = tf.fromPixels(image);

        // Crop the image so we're using the center square of the rectangular
        // webcam.
        const croppedImage = this.cropImage(webcamImage);

        // Expand the outer most dimension so we have a batch size of 1.
        const batchedImage = croppedImage.expandDims(0);

        // Normalize the image between -1 and 1. The image comes in between 0-255,
        // so we divide by 127 and subtract 1.
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });

    dsManager.addExample(img, 0);

    return dsManager;

};



  /**
   * Crops an image tensor so we get a square image with no white space.
   * @param {Tensor4D} img An input image Tensor to crop.
   */
  const cropImage = (img: tf.Tensor) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  };



const buildModel = (): tf.Sequential => {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({ filters: 128, activation: "relu", inputShape: [128, 128, 3], kernelSize: 3 }));
    model.add(tf.layers.maxPooling2d({  poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.conv2d({ filters: 256, activation: "relu", kernelSize: 3 }));
    model.add(tf.layers.maxPooling2d({  poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));

    return model;
};

const train = async (model: tf.Sequential, dsManager: dsManager ) => {
    model.compile( { optimizer: "rmsprop" , loss: "categoricalCrossentropy", metrics: ["accuracy"] } );
    const trainHistory = await model.fit(dsManager.xs, dsManager.ys, { epochs: 20 });
};

const prediction = async (model: tf.Sequential, input: tf.Tensor) => {
    const result = await model.predict(input);
    console.log(`${result}`);
    return result;
};

const saveModel = async (model: tf.Sequential) => {
    const saveResult = await model.save("./model/model1.json");
    console.log(saveResult);
};

const loadModel = async (path: string): Promise<tf.Model> => {
    const saveResult = await tf.loadModel(path);
    console.log(saveResult);
    return saveResult;
};

train(buildModel(), readImages());




