package deepnetts.examples.util;

import deepnetts.data.TabularDataSet;

import deepnetts.data.DataSets;
import java.io.File;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * TODO: add breast cancer, mnist  and other UCI stuff
 * @author Zoran
 */
public class ExampleDataSets {

    public static DataSet iris() throws IOException {
       return DataSets.readCsv("datasets/iris_data_normalised.txt", 4, 3);
    }

    public static DataSet xor() {
        DataSet dataSet = new TabularDataSet(2, 1);
        dataSet.setColumnNames(new String[] {"col1", "col2", "col3"});

        MLDataItem item1 = new TabularDataSet.Item(new float[] {0, 0}, new float[] {0});
        dataSet.add(item1);

        MLDataItem item2 = new TabularDataSet.Item(new float[] {0, 1}, new float[] {1});
        dataSet.add(item2);

        MLDataItem item3 = new TabularDataSet.Item(new float[] {1, 0}, new float[] {1});
        dataSet.add(item3);

        MLDataItem item4 = new TabularDataSet.Item(new float[] {1, 1}, new float[] {0});
        dataSet.add(item4);

        return dataSet;
    }


    /**
     * Download and unzip MNIST dataset
     *
     * @return path to downloaded data set
     * @throws IOException
     */
    public static Path downloadMnistDataSet() throws IOException { // print out message since it can take a while
        Path mnistPath = Paths.get("datasets", "mnist");
        File mnistFolder = mnistPath.toFile();
        System.out.println(String.format("Downloading and/or unpacking MNIST training set to: %s - this may take a while ( 44.9 MB )!", mnistFolder.getAbsolutePath()));
        if (!mnistFolder.exists()) {
            if (!mnistFolder.mkdirs()) {
                throw new IOException("Couldn't create temporary directories to download the Mnist training dataset.");
            }
        }

        // check if mnist zip  already exists  - don't download it again if its there
        Path zipPath = Paths.get(mnistPath.toString(), "dataset.zip");
        if (!Files.exists(zipPath)) {
            downloadZip("https://github.com/JavaVisRec/jsr381-examples-datasets/raw/master/mnist_training_data_png.zip", zipPath);
        }

        return mnistPath;
    }
    
    private static void downloadZip(String httpsURL, Path path) {
        URL url = null;
        try {
            File toFile = path.toFile();
            url = new URL(httpsURL);
            ReadableByteChannel rbc = Channels.newChannel(url.openStream());
            FileOutputStream fos = new FileOutputStream(toFile);
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
            fos.close();
            rbc.close();
            unzip(toFile);
        } catch (IOException ioException) {
            throw new RuntimeException(ioException);
        }
    }    
    
  private static void unzip(File fileToBeUnzipped) {
        File dir = new File(fileToBeUnzipped.getParent());

        if (!dir.exists())
            dir.mkdirs();

        try {
            ZipFile zipFile = new ZipFile(fileToBeUnzipped.getAbsoluteFile());
            Enumeration<?> enu = zipFile.entries();
            while (enu.hasMoreElements()) {
                ZipEntry zipEntry = (ZipEntry) enu.nextElement();
                String name = zipEntry.getName();

                if (name.contains(".DS_Store") || name.contains("__MACOSX"))
                    continue;

                File file = Paths.get(fileToBeUnzipped.getParent(), name).toFile();
                if (name.endsWith("/")) {
                    file.mkdirs();
                    continue;
                }
                if (file.exists()) {
                    continue;
                }

                File parent = file.getParentFile();
                if (parent != null) {
                    parent.mkdirs();
                }

                InputStream is = zipFile.getInputStream(zipEntry);
                FileOutputStream fos = new FileOutputStream(file);
                byte[] bytes = new byte[1024];
                int length;
                while ((length = is.read(bytes)) >= 0) {
                    fos.write(bytes, 0, length);
                }
                is.close();
                fos.close();
            }
            zipFile.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }    
}
