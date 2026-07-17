import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import math

class FeatureExtractor(nn.Module):
    def __init__(self, layer_indices=[5, 20, 35]):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.layer_indices = sorted(layer_indices)

    def forward(self, x):
        features = []

        x = self.feature_extractor._swish(
            self.feature_extractor._bn0(
                self.feature_extractor._conv_stem(x)
            )
        )

        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx in self.layer_indices:
                features.append(x)

            if len(features) == len(self.layer_indices):
                break

        output_feature = features[-1]
        return output_feature

class MultiFlowBackbone(nn.Module):
    def __init__(self):
        super(MultiFlowBackbone, self).__init__()
        self.feature_extractor = FeatureExtractor(layer_indices=[10, 20, 35])
        self.freeze_feature_extractor()

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

def test_model():
    model = MultiFlowBackbone()

    test_input = torch.randn(1, 3, 608, 608)

    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    return model, output

if __name__ == "__main__":
    model, output = test_model()

