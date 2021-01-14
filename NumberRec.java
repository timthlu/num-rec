import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

public class NumberRec {
	static int hiddenLayers = 10; //change if you want
	static int neuronsPerHidden = 100; //change if you want
	
	static Neuron[][] hidden = new Neuron[neuronsPerHidden][hiddenLayers];
	static Neuron[] output = new Neuron[10];
	
	static int firstLayerLen = 28*28;
	static double[] firstLayer = new double[firstLayerLen]; 
	
	public static void main(String[] args) throws IOException {
		
	//INITIALIZE NEURONS
		//first hidden layer have weights len = firstLayerLen
		for (int i = 0; i < neuronsPerHidden; i++) {
			double[] a = new double[firstLayerLen];
			
			for (int j = 0; j < firstLayerLen; j++) {
				a[j] = Math.random();
			}
			
			hidden[i][0] = new Neuron(a, 0);
		}
		
		//other hidden layers
		for (int i = 1; i < neuronsPerHidden; i++) {
			for (int j = 0; j < hiddenLayers; j++) {
				double[] a = new double[neuronsPerHidden];
				
				for (int k = 0; k < neuronsPerHidden; k++) {
					a[k] = Math.random();
				}
				
				hidden[i][j] = new Neuron(a, 0);
			}
		}
		
		//output layer
		for (int i = 0; i < 10; i++) {
			double[] a = new double[neuronsPerHidden];
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				a[j] = Math.random();
			}
			
			output[i] = new Neuron(a, 0);
		}
		
	//TRAIN - forward prop then back prop, can only train one by one
		int correct = 0;
		int total = 0;
		while (total == 0 || (correct/total) < 0.95) {
			total++;
			
			//initialize firstLayer from image
			File f = new File("image.jpg"); //SET PROPERLY *******************************
			BufferedImage image = ImageIO.read(f);
			int height = image.getHeight();
			int width = image.getWidth();
			
			double widthIncrement = width/28.0;
			double heightIncrement = height/28.0;
			
			//rgb = 65536*r + 256*g + b, rgb(r, g, b)
			//converting to grayscale: find r, g, b; then average them to find grayscale value
			
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					
					double sum = 0;
					for (int k = i*widthIncrement; k < (i+1)*widthIncrement; k++) { //x
						for (int l = j*heightIncrement; j < (j+1)*heightIncrement; j++) { //y
							int rgb = image.getRGB(k, l);
							int b = rgb % 256;
							int g = ((rgb-b)/256)%256;
							int r = (rgb - 256*g - b)/65536;
							
							sum += (r + g + b)/3.0;
							
						}
					}
					
					double average = sum/(widthIncrement*heightIncrement);
					firstLayer[j*28+i] = average;
				}
			}
			
			image.close();
			f.close();
			
			//testing
			forwardPropagate();
			int ans = getAns();
			
			int intendedAns = 0; //GET **********************
			
			if (ans == intendedAns) {
				correct++;
			} else {
				backwardPropagate();
			}
		}
		
	//Now fully trained, can take user input
		//**************************
		
		
		
	}
	
	public static void forwardPropagate() {
	//FORWARD PROPAGATE
		//first hidden layer
		for (int i = 0; i < neuronsPerHidden; i++) {
			Neuron n = hidden[i][0];
			n.value = n.bias;
			for (int j = 0; j < firstLayerLen; j++) {
				n.value += (firstLayer[j])*(n.weights[j]);
			}
			n.value = sigmoid(n.value);
		}
		
		//other hidden layers
		for (int i = 1; i < hiddenLayers; i++) {
			for (int j = 0; j < neuronsPerHidden; j++) {
				Neuron n = hidden[j][i];
				n.value = n.bias;
				for (int k = 0; k < neuronsPerHidden; k++) {
					n.value += (hidden[k][i-1].value)*(n.weights[k]);
				}
				n.value = sigmoid(n.value);
			}
		}
		
		//output layer
		for (int i = 0; i < 10; i++) {
			Neuron n = output[i];
			n.value = n.bias;
			for (int j = 0; j < neuronsPerHidden; j++) {
				n.value += (hidden[j][hiddenLayers-1].value)*(n.weights[j]);
			}
			n.value = sigmoid(n.value);
		}
	}
	
	public static double sigmoid(double d) { //chosen because it's easy to take a derivative of (vs relu and softmax)
		double denom = 1 + Math.pow(Math.E, -d);
		return (1/denom);
	}
	
	public static double derivSigmoid(double d) {
		return sigmoid(d)*(1 - sigmoid(d));
	}
	
	public static void backwardPropagate(double[] expectedOutput) {
	//calculate how much you need to change cost function in order to make it 0
		double costChange = 0;
		for (int i = 0; i < 10; i++) {
			costChange += Math.pow(output[i] - expectedOutput[i], 2);
		}
		costChange = -costChange; //gradient descent; go in opposite direction of slope.
		
	
	//get expected values
		double[][] expectedValues = new double[neuronsPerHidden][hiddenLayers]; //calculated from values, needed for weights and biases
		
		//initialize expectedValues
		for (int i = 0; i < hiddenLayers; i++) {
			for (int j = 0; j < neuronsPerHidden; j++) {
				expectedValues[j][i] = hidden[j][i].value;
			}
		}
		
		//output layer
		for (int i = 0; i < 10; i++) {
			Neuron cur = output[i];
			double curValue = cur.value;
			double curBias = cur.bias;
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				double prevValue = hidden[hiddenLayers-1][j].value;
				double curWeight = cur.weights[j];
				
				double change = costChange*curWeight*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedOutput[i]);
				expectedValues[j][neuronsPerHidden-1] += change;
			}
		}
		
		//hidden layers
		for (int i = hiddenLayers-1; i >= 1; i--) {
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				Neuron cur = hidden[j][i];
				double curValue = cur.value;
				double curBias = cur.bias;
				
				for (int k = 0; k < neuronsPerHidden; k++) {
					double prevValue = hidden[i-1][k].value;
					double curWeight = cur.weights[k];
					
					double change = costChange*curWeight*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][i]);
					expectedValues[k][i] += change;
				}
			}
		}
		
		//first hidden layer
		for (int i = 0; i < neuronsPerHidden; i++) {
			Neuron cur = output[i];
			double curValue = cur.Value;
			double curBias = cur.bias;
			
			for (int j = 0; j < firstLayerLen; j++) {
				double prevValue = firstLayerLen[j];
				double curWeight = cur.weights[j];
				
				double change = costChange*curWeight*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][0]);
				expectedValues[k][i] += change;
			}
		}
	
	//weights
		//output layer
		for (int i = 0; i < 10; i++) {
			Neuron cur = output[i];
			double curValue = cur.value;
			double curBias = cur.bias;
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				double prevValue = hidden[hiddenLayers-1][j].value;
				double curWeight = cur.weights[j];
				
				double change = costChange*prevValue*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedOutput[i]);
				cur.weights[j] += change;
				
			}
		}
	
		//hidden layers
		for (int i = hiddenLayers-1; i >= 1; i--) {
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				Neuron cur = hidden[j][i];
				double curValue = cur.value;
				double curBias = cur.bias;
				
				for (int k = 0; k < neuronsPerHidden; k++) {
					double prevValue = hidden[i-1][k].value;
					double curWeight = cur.weights[k];
					
					double change = costChange*prevValue*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][i]);
					cur.weights[k] += change;
				}
			}
		}
		
		//first hidden layer
		for (int i = 0; i < neuronsPerHidden; i++) {
			Neuron cur = output[i];
			double curValue = cur.Value;
			double curBias = cur.bias;
			
			for (int j = 0; j < firstLayerLen; j++) {
				double prevValue = firstLayerLen[j];
				double curWeight = cur.weights[j];
				
				double change = costChange*prevValue*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][0]);
				cur.weights[j] += change;
			}
		}
		
	//biases
		//output layer
		for (int i = 0; i < 10; i++) {
			Neuron cur = output[i];
			double curValue = cur.Value;
			double curBias = cur.bias;
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				double prevValue = hidden[hiddenLayers-1][j].value;
				double curWeight = cur.weights[j];
				
				double change = costChange*1*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedOutput[i]);
				curBias += change;
				
			}
		}
	
		//hidden layers
		for (int i = hiddenLayers-1; i >= 1; i--) {
			
			for (int j = 0; j < neuronsPerHidden; j++) {
				Neuron cur = hidden[j][i];
				double curValue = cur.value;
				double curBias = cur.bias;
				
				for (int k = 0; k < neuronsPerHidden; k++) {
					double prevValue = hidden[i-1][k].value;
					double curWeight = cur.weights[k];
					
					double change = costChange*1*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][i]);
					curBias += change;
				}
			}
		}
		
		//first hidden layer
		for (int i = 0; i < neuronsPerHidden; i++) {
			Neuron cur = output[i];
			double curValue = cur.Value;
			double curBias = cur.bias;
			
			for (int j = 0; j < firstLayerLen; j++) {
				double prevValue = firstLayerLen[j];
				double curWeight = cur.weights[j];
				
				double change = costChange*1*derivSigmoid(curWeight*prevValue + curBias)*2*(curValue - expectedValues[j][0]);
				curBias += change;
			}
		}
	}
	
	public static int getAns() {
		int max = 0;
		
		for (int i = 1; i < 10; i++) {
			if (output[max] < output[i]) {
				max = i;
			}
		}
		return max;
	}
	
	static class Neuron {
		double value;
		double[] weights;
		double bias;
		
		public Neuron(double[] weights, double bias) {
			this.weights = weights;
			this.bias = bias;
		}
	}
}