package deepnetts.examples.tensorflow.vggnet;

import deepnetts.util.TensorflowUtils;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Example for importing pre-trained VGGNet16 from Tensorflow.
 * VggNet is a large convolutional neural network trained with tensorflow on large dataset imagenet to recognize 1000 objects.
 * Original paper about VGGNet is available at https://arxiv.org/abs/1409.1556
 * This example requires minimum 7gb of memory to run run due to a  large  neural network.
 * 
 * For best performance use the following JVM switches: -Xms7g  -XX:MaxInlineSize=50
 * -Xms3g starts jvm with min 3g of memory
 * -XX:MaxInlineSize=50 sets size for inlining methods which improves performance significantly
 */
public class VggNetImport {
    public static void main(String[] args) throws IOException {
        
        // step 1: create the vggnet neural network architecture that will import weights
        ConvolutionalNetwork vggNet16 = ConvolutionalNetwork.builder()
                .addInputLayer(224, 224, 3)
                .addConvolutionalLayer(64, Filter.ofSize(3))
                .addConvolutionalLayer(64, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)

                .addConvolutionalLayer(128, Filter.ofSize(3))
                .addConvolutionalLayer(128, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)  
                
                .addConvolutionalLayer(256, Filter.ofSize(3))
                .addConvolutionalLayer(256, Filter.ofSize(3))
                .addConvolutionalLayer(256, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2) 

                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)    
                
                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addConvolutionalLayer(512, Filter.ofSize(3))
                .addMaxPoolingLayer(2, 2)                 
                
                .addFullyConnectedLayer(4096)
                .addFullyConnectedLayer(4096)
                .addOutputLayer(1000, ActivationType.SOFTMAX)
                .hiddenActivationFunction(ActivationType.RELU)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        // step 2; download pretrained network parameters(weights and biases) and import them in the network above
        downloadVggNet16WeightsFile();
        String userHomeDir = System.getProperty("user.home");
        System.out.println("Importing pretrained weights from tensorflow model..."); 
        TensorflowUtils.importWeights(vggNet16, userHomeDir + "/.deepnetts/vgg16_imagenet_weights.txt");// ovo neka bude utility funkcija koju ubacim i u builder
        
        // load output labels from csv file and set them as neural network's outputs
        try {            
            String labelsStr = Files.readString(Path.of("datasets/vggnet_labels.csv")); // ove labele najbolje serijalizovati u net prilikom importa - svakako ih treba ubaciti u labels outputa!!
            String[] labels = labelsStr.split(",");        
            vggNet16.setOutputLabels(labels);
        } catch (IOException ex) {
            Logger.getLogger(VggNet16.class.getName()).log(Level.SEVERE, null, ex);
        }
                
       // save the imported network to file
       vggNet16.save(userHomeDir+"/.deepnetts/vggnet16.dnet");     
       System.out.println("Imported network ands saved VggNet successfully.");
    }
    
    /**
     * Downloads vggnet16 weights file if it does not exist in .deepnetts directory
     */
    public static void downloadVggNet16WeightsFile() {
        String userHomeDir = System.getProperty("user.home");
        Path deepNettsDir = Paths.get(userHomeDir, ".deepnetts");
                
        if (!Files.exists(deepNettsDir)) {
            deepNettsDir.toFile().mkdir();
        }
        
        File file = new File(deepNettsDir + "/vgg16_imagenet_weights.txt");
        if (!file.exists()) { 
            System.out.println("VggNet pre-trained weights file is not available in local deepnetts dir, downloading it. It will take some time depending on the connection speed (file size: 2GB)");
            try {
                File toFile = new File(deepNettsDir+"/vgg16_imagenet_weights.txt");
                URL url = new URL("https://dl.dropboxusercontent.com/s/oyab8qtzoaiivpm/vgg16_imagenet_weights.txt?dl=0");
                ReadableByteChannel rbc = Channels.newChannel(url.openStream());
                FileOutputStream fos = new FileOutputStream(toFile);
                fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
                fos.close();
                rbc.close();
            } catch (MalformedURLException ex) {
                Logger.getLogger(VggNetImport.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(VggNetImport.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
     
}