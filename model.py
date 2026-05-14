import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, eo_feat, sar_feat):

        diff = torch.abs(eo_feat - sar_feat)

        x = torch.cat([
            eo_feat,
            sar_feat,
            diff
        ], dim=1)

        return self.fuse(x)


class DualEncoderFusionNet(nn.Module):

    def __init__(
        self,
        encoder_name="efficientnet-b2",
        pretrained=True
    ):
        super().__init__()

        self.eo_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet" if pretrained else None
        )

        self.sar_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights="imagenet" if pretrained else None
        )

        encoder_channels = self.eo_encoder.out_channels

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(ch)
            for ch in encoder_channels
        ])

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            attention_type="scse"
        )

        self.segmentation_head = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, x):

        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        sar = sar.repeat(1, 3, 1, 1)

        eo_features = self.eo_encoder(eo)
        sar_features = self.sar_encoder(sar)

        fused_features = []

        for eo_f, sar_f, fusion in zip(
            eo_features,
            sar_features,
            self.fusion_blocks
        ):
            fused_features.append(
                fusion(eo_f, sar_f)
            )

        decoder_output = self.decoder(fused_features)

        masks = self.segmentation_head(decoder_output)

        return masks


if __name__ == "__main__":

    model = DualEncoderFusionNet(pretrained=True)

    x = torch.randn(2, 4, 384, 384)

    out = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {total:,}")