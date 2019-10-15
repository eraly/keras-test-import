package org.deeplearning4j.examples.modelimport.keras.basic;

import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;

/**
 * Created by susaneraly on 9/9/19.
 */
public class TestImport {

    public static void main(String[] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        final String MODEL_PATH = "/Users/susaneraly/SKYMIND/user_model.h5";
        final String IMAGE_PATH = new ClassPathResource("random.jpg").getFile().getAbsolutePath();
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(MODEL_PATH);

        BufferedImage image = ImageIO.read(new File(IMAGE_PATH));
        NativeImageLoader loader = new NativeImageLoader(224,224,3, new ColorConversionTransform(COLOR_BGR2RGB));
        INDArray input1 = loader.asMatrix(image);
        input1 = input1.divi(127.5f).subi(1.f);
        input1 = input1.reshape(1,3,224,224);


        System.out.println(model.output(input1)[0]);
        //should be array [[0.99686974 0.00313022]]
    }
}
