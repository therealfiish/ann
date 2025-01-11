#include <vector>
#include <iostream>

class Neuron {};
typedef std::vector<Neuron> Layer;

class Net {
public:
    Net(const std::vector<unsigned> &topology);
    void feedFwd(const std::vector<double> &inputVals) {};
    void backProp(const std::vector<double> &targetVals) {};
    void getRel(std::vector<double> resultVals) const {};

private:
    std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned nLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < nLayers; layerNum++)
    {
        m_layers.push_back(Layer());

        //new layer made, now need neurons in layers. also add bias neuron to the layers
        // adding bias neuron by utilzing the <= instead of just <
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
        {
            m_layers.back().push_back(Neuron());
            std::cout << "neuron made" << std::endl;
        }
    }
}
int main() {

    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net nNet(topology);

    std::vector<double> inputVals;
    nNet.feedFwd(inputVals);

    std::vector<double> targetVals;
    nNet.backProp(targetVals);

    std::vector<double> resultVals;
    nNet.getRel(resultVals);

    return 0;
}

