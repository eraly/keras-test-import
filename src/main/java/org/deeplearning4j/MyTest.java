package org.deeplearning4j;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by susaneraly on 2/11/20.
 */
@Slf4j
public class MyTest {

    public static void main(String[] args) throws Exception {
        new MyTest().runTest();
    }

    public void runTest() throws Exception{
        String model_path = new ClassPathResource("model_toy.h5").getFile().getAbsolutePath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(model_path, false);
        List<String> inputFilePrefix = new ArrayList<>();
        inputFilePrefix.add("test_toy_random");
        inputFilePrefix.add("test_toy");

        for (String eachIn: inputFilePrefix) {
            System.out.println("Running input with prefix " + eachIn);
            INDArray input = Nd4j.createFromNpyFile(new ClassPathResource(eachIn + ".npy").getFile());
            INDArray expectedOut = Nd4j.createFromNpyFile(new ClassPathResource(eachIn + "_out.npy").getFile());
            INDArray actual = model.output(input);
            System.out.println("EXPECTED:\n" + expectedOut);
            System.out.println("ACTUAL:\n" + actual);
            Assert.isTrue(expectedOut.equalsWithEps(actual,1e-4));
        }
    }


    /*
    public static void compareOutputs(int layerNum, MultiLayerNetwork model, String prefix, INDArray input) throws Exception {

        if (layerNum == 0) {
            System.out.println(model.summary());
            System.out.println("This is the input shape to the model: \n" + ArrayUtils.toString(input.shape()));
            //return;
        }
        INDArray actual = model.feedForwardToLayer(layerNum, input).get(layerNum + 1);
        String outFileName = model.getLayer(layerNum).conf().getLayer().getLayerName() + ".npy";
        if(!prefix.equals("")) outFileName = prefix + "_" + outFileName;
        System.out.println(outFileName + " loading...");
        INDArray expected = Nd4j.createFromNpyFile(new ClassPathResource(outFileName).getFile());

        System.out.println("ACTUAL...");
        System.out.println("Actual array shape: " + ArrayUtils.toString(actual.shape()));
        if (layerNum > 0) System.out.println(actual);
        //System.out.println(actual.maxNumber());
        //System.out.println(actual.minNumber());
        System.out.println(actual);
        System.out.println("EXPECTED...");
        System.out.println("Expected array shape: " + ArrayUtils.toString(expected.shape()));
        if(layerNum > 0) System.out.println(expected);
        //System.out.println(expected.maxNumber());
        //System.out.println(expected.minNumber());
        System.out.println(expected);
        //System.out.println(expected.div(actual).maxNumber());
        //System.out.println(expected.div(actual).minNumber());
        if (layerNum == 7)
            System.out.println("This is output of the model: \n" + model.output(input));
    }
    */
}
