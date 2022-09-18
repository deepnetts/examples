package deepnetts.examples.tensorflow.vggnet;

import deepnetts.core.DeepNetts;
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
 * Example how to use pre-trained saved VggNet16 imported from Tensorflow for inference.
 * VggNet is a large convolutional neural network trained with tensorflow on large dataset imagenet to recognize 1000 objects.
 * An example how to import vgg net neural network  from Tensorflow is given in VggNetImport.
 * Original paper about VGGNet is available at https://arxiv.org/abs/1409.1556
 * This example requires minimum 3gb of memory to run due to a  large  neural network.
 * 
 * For best performance use the following JVM switches to run this example: -Xms3g  -XX:MaxInlineSize=50
 * -Xms3g starts jvm with min 3g of memory
 * -XX:MaxInlineSize=50 sets size for inlining methods which improves performance significantly
 * 
 */
public class VggNetInference {
    
    public static void main(String[] args) throws IOException, ClassNotFoundException {    
        // vgg net file {user.home}/.deepnetts/vggnet16.dnet
        String userHomeDir = System.getProperty("user.home");  
        String deepNettsDir = userHomeDir + "/.deepnetts";
        String vggNetFile = deepNettsDir + "/" + "vggnet16.dnet";
        
        // download pre-trained saved vggnet16 from tensorflow into local {user.home}/.deepnetts dir, if it does now allready exist there
        downloadIfNotExists(vggNetFile, "https://drive.google.com/uc?id=1Y_Uy-6TWjdRC-onNHPbNBbf9BEel4nb7&export=download&confirm=t&uuid=07f28a6f-5244-4ec1-9c4f-e5727d5d1eb9");
        // https://drive.google.com/uc?id=1Y_Uy-6TWjdRC-onNHPbNBbf9BEel4nb7&export=download
        //https://drive.google.com/uc?id=1Y_Uy-6TWjdRC-onNHPbNBbf9BEel4nb7&export=download&confirm=t&uuid=07f28a6f-5244-4ec1-9c4f-e5727d5d1eb9
        // https://drive.google.com/uc?id=1Y_Uy-6TWjdRC-onNHPbNBbf9BEel4nb7&export=download&confirm=t&uuid=07f28a6f-5244-4ec1-9c4f-e5727d5d1eb9
        //* https://www.marstranslation.com/blog/how-to-skip-google-drive-virus-scan-warning-about-large-files

        // create an instance of trained VGGNet16 from file 
        VggNet16 neuralNetwork = VggNet16.fromFile(vggNetFile);     
        
        // guess/predict a label for the given image (specified as path to the image)
        String label = neuralNetwork.guessLabel("datasets/test_vgg/airplane.jpg"); // change this path to an image to test other images/objects           
       
        // print predicted label
        System.out.println("This image contains: " + label);
        
        // shutdown the deep netts thread pool
        DeepNetts.shutdown();
    }

    /**
     * Checks if specifed file exists, and if not downloads it from specified url to {user.home}/.deepnetts
     * @param fileName
     * @param urlStr 
     */
    static void downloadIfNotExists(String fileName, String urlStr) {
        String userHomeDir = System.getProperty("user.home");
        Path deepNettsDir = Paths.get(userHomeDir, ".deepnetts");
                
        if (!Files.exists(deepNettsDir)) {
            deepNettsDir.toFile().mkdir();
        }
        
        File file = new File(deepNettsDir + "/vggnet16.dnet");
        if (!file.exists()) {
            System.out.println("VggNet pre-trained network file not available on local disk, downloading it. It will take some time depending on the connection speed (file size: 2.6GB)");
            try {
                URL url = new URL(urlStr); // gde cu je uploadovati - gdrive?? stavi link ovde
                ReadableByteChannel rbc = Channels.newChannel(url.openStream());
                FileOutputStream fos = new FileOutputStream(file);
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
