import torch
import torchvision
from omegaconf import DictConfig


class Model(torchvision.models.ResNet):
    def __init__(self, config: DictConfig) -> None:
        block_name = config.model.backbone.resnet_block
        if block_name == "basic":
            block = torchvision.models.resnet.BasicBlock
        elif block_name == "bottleneck":
            block = torchvision.models.resnet.Bottleneck
        else:
            raise ValueError
        layers = [*config.model.backbone.resnet_layers, 1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        pretrained_name = config.model.backbone.pretrained
        if pretrained_name:
            # torchvision removed `model_urls` in 0.15. IMAGENET1K_V1 points
            # to the same weight files the old URLs did.
            weights = torchvision.models.get_model_weights(pretrained_name).IMAGENET1K_V1
            self.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            # The torchvision pretrained weights expect RGB input, but
            # this repository feeds images in BGR channel order, so
            # reverse the channel order of the first convolutional
            # layer's weights.
            module = self.conv1
            module.weight.data = module.weight.data[:, [2, 1, 0]]

        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
