#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
// class TrainingData
// {
// public:
//     TrainingData(const std::string filename);
//     bool isEof(void)
//     {
//         return m_trainingDataFile.eof();
//     }
//     void getTopology(std::vector<unsigned> &topology);
//
//     // Returns the number of input values read from the file:
//     unsigned getNextInputs(std::vector<double> &inputVals);
//     unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
//
// private:
//     std::ifstream m_trainingDataFile;
// };
//
// void TrainingData::getTopology(std::vector<unsigned> &topology)
// {
//     std::string line;
//     std::string label;
//
//     getline(m_trainingDataFile, line);
//     std::stringstream ss(line);
//     ss >> label;
//     if(this->isEof() || label.compare("topology:") != 0)
//     {
//         abort();
//     }
//
//     while(!ss.eof())
//     {
//         unsigned n;
//         ss >> n;
//         topology.push_back(n);
//     }
//     return;
// }
//
// TrainingData::TrainingData(const std::string filename)
// {
//     m_trainingDataFile.open(filename.c_str());
// }
//
//
// unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
// {
//     inputVals.clear();
//
//     std::string line;
//     getline(m_trainingDataFile, line);
//     std::stringstream ss(line);
//
//     std::string label;
//     ss >> label;
//     if (label.compare("in:") == 0) {
//         double oneValue;
//         while (ss >> oneValue) {
//             inputVals.push_back(oneValue);
//         }
//     }
//
//     return inputVals.size();
// }
//
// unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
// {
//     targetOutputVals.clear();
//
//     std::string line;
//     getline(m_trainingDataFile, line);
//     std::stringstream ss(line);
//
//     std::string label;
//     ss>> label;
//     if (label.compare("out:") == 0) {
//         double oneValue;
//         while (ss >> oneValue) {
//             targetOutputVals.push_back(oneValue);
//         }
//     }
//
//     return targetOutputVals.size();
// }
class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};
class Neuron
{
public:
    Neuron(unsigned nOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val;}
    double getOutputVal() const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    static double randomWeight() { return rand() / double(RAND_MAX); }
    unsigned m_myIndex;
    double m_gradient;
    double sumDOW(const Layer &nextLayer);
    static double alpha; // 0.0 -> n, multiplier of last weight chance (momentum)
    static double eta; // 0.0 -> 1.0, overall net training rate
};


double Neuron::eta = 0.2; //overall net learning rate
double Neuron::alpha = 0.7; //momentum - multiplier of last delWeight (0.0 -> n)

Neuron::Neuron(unsigned nOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < nOutputs; c++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}
void Neuron::updateInputWeights(Layer& prevLayer)
{
     //weights to be updated in connection container in the neurons in the preceding layer.

    for (unsigned n = 0; n < prevLayer.size(); n++)
    {
        Neuron &neuron = prevLayer[n];
        double oldDelWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDelWeight =
            // individual input, magnified by gradient and train rate:
            eta
            * neuron.getOutputVal()
            * m_gradient
            // add momentum - fraction of previous del weight
            + alpha
            * oldDelWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDelWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDelWeight;

    }
}
double Neuron::sumDOW(const Layer& nextLayer)
{
    double sum = 0.0;
     //sum contributions of errors at nodes we feed
    for (unsigned n = 0; n <nextLayer.size() - 1; n++)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}
void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow*Neuron::transferFunctionDerivative(m_outputVal);
}
void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta*Neuron::transferFunctionDerivative(m_outputVal);
}
double Neuron::transferFunction(double x)
{
    //tanh - output range [-1 to 1]
    return tanh(x);
}
double Neuron::transferFunctionDerivative(double x)
{
    //tanh derivative
    return 1.0 - x*x;
}
void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    //Sum previous layers outputs (our inputs)
    //Include bias node from previous layer

    for (unsigned n = 0; n < prevLayer.size(); n++)
    {
        sum += prevLayer[n].getOutputVal() *
            prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

class Net {
public:
    Net (const std::vector<unsigned> &topology);
    void feedFwd(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getRel(std::vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageErr; }

private:
    std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_err;
    double m_recentAverageErr;
    static double m_recentAverageSmoothingFactor;
};
double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over
void Net::getRel(std::vector<double> &resultVals) const
{
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size()-1; n++)
    {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}
void Net::backProp(const std::vector<double> &targetVals)
{
    // Calculate overal net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_err = 0.0;

    for(unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_err += delta * delta;
    }
    m_err /= outputLayer.size() - 1;
    m_err = sqrt(m_err);

    m_recentAverageErr =
        (m_recentAverageErr * m_recentAverageSmoothingFactor + m_err)
        / (m_recentAverageSmoothingFactor + 1.0);
    // Calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer.size(); ++n)
        {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights
    for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for(unsigned n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}
void Net::feedFwd(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    //assign the input vals into input neurons
    for (unsigned i = 0; i < inputVals.size(); i++)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //fwd propogate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
    {
        Layer &prevLayer = m_layers[layerNum-1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
 }
Net::Net(const std::vector<unsigned> &topology)
{
    unsigned nLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < nLayers; layerNum++)
    {
        m_layers.push_back(Layer());
        unsigned nOutputs = (layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1]);

        //new layer made, now need neurons in layers. also add bias neuron to the layers
        // adding bias neuron by utilizing the <= instead of just <
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
        {
            m_layers.back().push_back(Neuron(nOutputs, neuronNum));
        }

        m_layers.back().back().setOutputVal(1.0);
    }

    // Force the bias node's output value to 1.0. It's the last neuron created above
    m_layers.back().back().setOutputVal(1);
}
void showVectorVals(std::string label, std::vector<double> &v, std::ofstream &out)
{
    out << label << " ";
    for(unsigned i = 0; i < v.size(); ++i)
    {
        out << v[i] << " ";
    }
    out << std::endl;
}

int main() {
    std::ofstream outFile("out.txt", std::ios::out);
    std::ifstream file("XORs.csv");
    std::string line;
    std::vector<std::vector<double>> inputs, outputs;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> inputRow, outputRow;

        // Read inputs
        while (std::getline(ss, value, ',')) {
            inputRow.push_back(std::stod(value));
        }

        // Last value is the output
        outputRow.push_back(inputRow.back());
        inputRow.pop_back();

        inputs.push_back(inputRow);
        outputs.push_back(outputRow);
    }

    std::vector<unsigned> topology = {2, 4, 1};
    Net myNet(topology);

    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    for (int i = 0; i < inputs.size(); ++i) {
        ++trainingPass;



        outFile << std::endl << "Pass " << trainingPass << ":\n";

        // Get new input data and feed it forward:
        inputVals = inputs[i];
        showVectorVals("Inputs:", inputVals, outFile);
        myNet.feedFwd(inputVals);

        // Collect the net's actual results:
        myNet.getRel(resultVals);
        showVectorVals("Outputs:", resultVals, outFile);

        // Train the net what the outputs should have been:
        targetVals = outputs[i];
        showVectorVals("Targets:", targetVals, outFile);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent
        outFile << "Net recent average error: "
                  << myNet.getRecentAverageError() << std::endl;
        outFile << "----------------------\n";
    }

}