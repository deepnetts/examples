package deepnetts.examples.tensorflow.vggnet;

import deepnetts.core.DeepNetts;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

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
        DeepNetts.getInstance().setUseCuda(true);
        // vgg net file {user.home}/.deepnetts/vggnet16.dnet
        String userHomeDir = System.getProperty("user.home");  
        String deepNettsDir = userHomeDir + "/.deepnetts";
        String vggNetFile = deepNettsDir + "/vggnet16.dnet";

        // download and unpack pre-trained vggnet16 into local {user.home}/.deepnetts dir (if it does now allready exist there)
        downloadIfNotExists(vggNetFile, "https://www.dropbox.com/scl/fi/tnwww1p9ie5wttuglt3m7/vggnet16_3.1.0.zip?rlkey=35frfi498gj6nm693rzge8apb&dl=1");

        // create an instance of trained VGGNet16 from file 
        VggNet16 neuralNetwork = VggNet16.fromFile(vggNetFile); 

        // load and preprocess an image
        VggNet16InputImage vggInputImage = new VggNet16InputImage("datasets/test_vgg/airplane.jpg");     
                
        // guess/predict a label for the given image (specified as path to the image)
        // String label = neuralNetwork.guessLabel("datasets/test_vgg/airplane.jpg"); // change this path to an image to test other images/objects           
        // run once to warmup JVM
        String label = neuralNetwork.guessLabel(vggInputImage);
        
        long startTime = System.currentTimeMillis(); 
        label = neuralNetwork.guessLabel(vggInputImage);              
     //   label = neuralNetwork.guessLabel("datasets/test_vgg/airplane.jpg");     
        long stopTime = System.currentTimeMillis();
        
        // print predicted label and inference time
        System.out.println("This image contains: " + label + " time:" + (stopTime-startTime));
                
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
        File zipFile = new File(deepNettsDir + "/vggnet16_3.1.0.zip");
        File file = new File(deepNettsDir + "/vggnet16.dnet");
        if (!file.exists()) {
            System.out.println("VggNet pre-trained network file not available on local disk, downloading it. It will take some time depending on the connection speed (file size: 2.6GB)");
            try {                
                URL url = new URL(urlStr); 
                ReadableByteChannel rbc = Channels.newChannel(url.openStream());
                FileOutputStream fos = new FileOutputStream(zipFile);
                fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
                fos.close();
                rbc.close();                
                unzip(zipFile);
            } catch (MalformedURLException ex) {
                Logger.getLogger(VggNetImport.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(VggNetImport.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
     private static void unzip(File fileToUnzip) {
        try ( ZipFile zipFile = new ZipFile(fileToUnzip.getAbsoluteFile())) {

            Enumeration<?> enu = zipFile.entries();
            while (enu.hasMoreElements()) {
                ZipEntry zipEntry = (ZipEntry) enu.nextElement();
                String name = zipEntry.getName();
                File file = Paths.get(fileToUnzip.getParent(), name).toFile();

                InputStream is = zipFile.getInputStream(zipEntry);
                FileOutputStream fos = new FileOutputStream(file);
                byte[] bytes = new byte[4096];
                int length;
                while ((length = is.read(bytes)) >= 0) {
                    fos.write(bytes, 0, length);
                }
                is.close();
                fos.close();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }      
}

// This image contains:  airliner time:10687
