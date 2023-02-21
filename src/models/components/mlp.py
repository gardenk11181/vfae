import torch
from torch import nn

class SingleLayerPerceptron(nn.Module):
    def __init__(
            self,
            input_size: int = 256,
            output_size: int = 256
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU()
                )

    def forward(self, x): #(B, I)
        assert len(x.size()) == 2
        output = self.model(x)
        
        return output

class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        output_size: int = 10,
        num_hiddens: int = 0
    ):
        super().__init__()

        self.model = nn.ModuleList([])
        if num_hiddens == 0:
            layer = SingleLayerPerceptron(input_size, output_size)
            self.model.append(layer)
        else:
            first_layer = SingleLayerPerceptron(input_size, hidden_size)
            final_layer = SingleLayerPerceptron(hidden_size, output_size)
            self.model.append(first_layer)
            for _ in range(num_hiddens-1):
                middle_layer = SingleLayerPerceptron(hidden_size, hidden_size)
                self.model.append(middle_layer)
            self.model.append(final_layer)

    def forward(self, x): #(B, I)
        assert len(x.size()) == 2
        for layer in self.model:
            x = layer(x)

        return x

if __name__ == "__main__":
    a = MultiLayerPerceptron()
    b = MultiLayerPerceptron(num_hiddens=1)
    c = MultiLayerPerceptron(num_hiddens=2)
    x = torch.rand((10,784))
    assert a(x).size() == (10, 10), 'Shape incorrect'
    assert b(x).size() == (10, 10), 'Shape incorrect'
    assert c(x).size() == (10, 10), 'Shape incorrect'
