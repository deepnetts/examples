package deepnetts.examples.tensorflow;

import deepnetts.util.TensorflowUtils;
import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.TabularDataSet;
import deepnetts.data.TrainTestPair;
import deepnetts.data.norm.MaxScaler;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.Tensor;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * Example how to import weights of feed forward neural network trained with tensorflow.
 * This example imports small feed forward neural network trained for iris classification dataset.
 */
public class ImportFFNWeights {
    public static void main(String[] args) throws IOException {
                   
        // step 1: create the network that will import weights
        FeedForwardNetwork network = FeedForwardNetwork.builder()
                                                        .addInputLayer(4)
                                                        .addFullyConnectedLayer(16, ActivationType.RELU)
                                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                                        .lossFunction(LossType.CROSS_ENTROPY)
                                                        .build();
       
        // step 2; read exported weights and biases from file and set the weights in network above
        TensorflowUtils.importWeights(network, "iris_exported_weights.txt");
        
          
        // step 3: test the network with imported weights with dataset
        TabularDataSet<?> dataSet = DataSets.readCsv("iris-flowers.csv", 4, 3, true, ",");
        TrainTestPair trainTest = DataSets.trainTestSplit(dataSet, 0.65);

        // normalize data using max normalization
        MaxScaler scaler = new MaxScaler(trainTest.getTrainingeSet());
        scaler.apply(trainTest.getTrainingeSet());   
        scaler.apply(trainTest.getTestSet());  
             
         //evaluate network with the test set
        EvaluationMetrics evalResult = network.test(trainTest.getTestSet());  
        System.out.println(evalResult);
        
        // shutdown all threads
        DeepNetts.shutdown();        
    }
         
}

/**
 * 
 *     static void testLayerOutputs(FeedForwardNetwork network, Tensor input) {
        network.setInput(input);
        for(AbstractLayer layer : network.getLayers()) {
            System.out.println("Layer output: " + layer.getOutputs());
        }
    }
 * 
 *  Input tensor: 0.8607595; 0.71428573; 0.82089555; 0.84
 *weights:
 * ovde bi export morao da bude drugaciji!!!: 4 kolone i 16 weights moram da resim export!!!! razdvoj logicki i fizicki layout tu je neki zbun 
 * export je dobar ali ovde bi ga trebalo transposovati ; pre je radio kao keras ali sad sam g aprilagodio za cuda 
 * 0.47658885, -0.31355044, -0.026475787, 0.22104949, 0.30110404, -0.2593418, 0.06375398, -0.12606788, -0.025115073, 0.21775338, 0.37582147, 0.13589253, 0.41430777, 0.67883307, -0.27840763, -0.020275522, 
-0.35043257, 0.014319219, -0.49481595, -0.3181225, -0.47223595, -0.5041512, 0.049869128, -0.33963716, -0.34680718, -0.1970396, -0.10318054, 0.0012943574, 0.3333673, 0.773443, -0.23221503, -0.3061301, 
-0.5287788, 0.6687289, 0.4604566, -0.2498621, 0.4403816, -0.12953287, 0.37282756, -0.45931724, -0.3996821, 0.76724565, -0.53120756, 0.035402548, -0.600347, -0.12794437, 0.34277698, 0.106488064, 
-0.10627097, 0.74074614, -0.27390948, -0.38559642, 0.6399515, -0.086452276, 0.6468567, -0.15649274, 0.22572094, 0.8005784, 0.2210847, 0.66135234, -0.31667164, -0.12122251, 0.69409406, 0.63822216, 

biases:0.0, -0.009311829, 0.0, 0.0, 0.041174512, 0.0, -0.21141204, 0.0, 0.0, -0.20771238, -0.10788461, -0.1391352, 0.36317667, 0.32892445, -0.16269727, -0.1683358

 *  Output : 0.3502115; 0.0; 0.19061476; 0.0; 0.18675765; 0.0; 0.85791016; 0.66297954; 0.16443107; 0.0; 0.36268452; 0.0; 0.60918224; 0.0; 0.14363204; 0.42925805
 * 
 * formula je:
 * 
 * out = W * I + B
 * 
 * Keras dense layer https://keras.io/api/layers/core_layers/dense/
 * output = activation(dot(input, kernel) + bias) - a keras obrce u dotu
 * posto sam im zamenio mesta morao bih da transponujem weights matricu da bi radila isto mislim
 * 
 * u fc layeru je y = f(W*x + b)  i ovako je uradjeno da bi bilo isto kao cublasSgemv https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemv
 * kako transposovati tensor? 2d matricu?
 * naso sam ga to je caka. tf weights tenzor mora biti transponovan za deep netts (zato jer je sad optimizovan za cuda)
 * zapravo on ih kreira isto kao u fajlu a treba da ih transponuje zbog cuda, ili da ih cuda transponuje a oni da ostanu isti....razmisli o tome
 */
