package org.deeplearning4j;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

public class NERTests {
    public static void main(String[] args) throws Exception {
        String [] modelList = new String[] {"word","casing","char"};
        for (int i=0; i<modelList.length;i++) {
            String modelName = modelList[i];
            //Assert.isTrue(new NERTests().runModel("word"));
            if (new NERTests().runModel(modelName)) {
                System.out.println(modelName + "passes import");
            }
            else {
                //throws exception
                System.out.println(modelName + "fails import");
            }
        }
    }

    public boolean runModel(String prefix) throws Exception {
        ComputationGraph model = getModel(prefix + "_model.h5");
        INDArray input = getInput(prefix);
        INDArray expectedOut = getOutput(prefix);
        INDArray actual = model.output(new INDArray[] {input})[0];
        System.out.println("EXPECTED:\n" + expectedOut);
        System.out.println("ACTUAL:\n" + actual);
        return expectedOut.equalsWithEps(actual,1e-4);
    }

    public ComputationGraph getModel(String modelName) throws Exception {
        String modelPath = new ClassPathResource("NER/" + modelName).getFile().getAbsolutePath();
        return KerasModelImport.importKerasModelAndWeights(modelPath,false);
    }

    public INDArray getInput(String prefix) throws IOException {
        return getArray(prefix + "_input.npy");
    }

    public INDArray getOutput(String prefix) throws IOException {
        return getArray(prefix + "_output.npy");
    }

    public INDArray getArray(String fileName) throws IOException {
        return Nd4j.createFromNpyFile(new ClassPathResource("NER/" + fileName).getFile());
    }
}
