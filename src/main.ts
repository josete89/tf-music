import "@tensorflow/tfjs-node";
import * as tf from "@tensorflow/tfjs";
import dsManager from "./ds.manager";
import DsManager from "./ds.manager";
import fs, { read } from "fs";
import { buffer } from "@tensorflow/tfjs";
const jpeg = require('jpeg-js');

tf.setBackend("tensorflow");


const readImages = (): DsManager => {
    const dsManager = new DsManager(3);

    const directoryInfo = [{path: "/Users/alcaljos/test-data/stan smith", label: 0},
                           {path: "/Users/alcaljos/test-data/nmd", label: 1},
                           {path: "/Users/alcaljos/test-data/superstar", label: 2}];

    const files = (path: string): Buffer[] => {
        return fs.readdirSync(directoryInfo[0].path).map((file) => {
            if (file == ".DS_Store") { return; }
            console.log(file);
            const path = directoryInfo[0].path;
            const data =  fs.readFileSync(`${path}/${file}`);
            const rawImageData = jpeg.decode(data, true);
            return rawImageData.data;
        });
    };

    const convertImgToTensor = (label: number) => (data: Buffer) => {
        const tensor = tf.tidy( () => {
            const t = tf.tensor3d(data, [128, 128, 4], "int32");
            const batchedImage = t.expandDims(0);
            batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
            return batchedImage;
        });
        dsManager.addExample(tensor, 0);
    };

    directoryInfo.forEach( (info) => {
        const transform = convertImgToTensor(info.label);
        files(info.path).forEach(transform);
    });

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
    model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

    return model;
};

const train = async (model: tf.Sequential, dsManager: dsManager ) => {
    model.compile( { optimizer: "rmsprop" , loss: "categoricalCrossentropy", metrics: ["accuracy", "val_acc"] } );
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



