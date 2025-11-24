import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Channel attention
        self.ca_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.ca_sig  = nn.Sigmoid()

        # Spatial attention
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sa_sig  = nn.Sigmoid()

        # Fuse after concat
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)

    def forward(self, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        F = r + d                               # [B,C,H,W]

        ca = F.mean(dim=(2,3), keepdim=True)    # [B,C,1,1]
        ca = self.ca_sig(self.ca_conv(ca))

        sa = F.mean(dim=1, keepdim=True)        # [B,1,H,W]
        sa = self.sa_sig(self.sa_conv(sa))

        r_w = r * ca * sa
        d_w = d * ca * sa

        out = torch.cat([r_w, d_w], dim=1)      # [B,2C,H,W]
        out = self.fuse(out)                            # [B,C,H,W]
        return out

class FusionBackbone(nn.Module):
    def __init__(self, blocks_per_stage=(2,2,2,2), ch=(64,128,256,512)):
        super().__init__()
        # stage modules
        self.stage1 = self._make_stage(ch[0], ch[1], blocks_per_stage[1], stride=2)
        self.stage2 = self._make_stage(ch[1], ch[2], blocks_per_stage[2], stride=2)
        self.stage3 = self._make_stage(ch[2], ch[3], blocks_per_stage[3], stride=2)

    def _make_stage(self, in_ch, out_ch, num_blocks, stride=1):
        return nn.Sequential(
            ResnetBlock(in_ch, out_ch, stride=stride),
            *[ResnetBlock(out_ch, out_ch, 1) for _ in range(num_blocks-1)]
        )

    def forward(self, C_list):
        F1 = self.stage1(C_list[0])
        F2 = self.stage2(F1 + C_list[1])
        F3 = self.stage3(F2 + C_list[2])
        F4 = F3 + C_list[3]
        return F4

class RGBDFusionCAB(nn.Module):
    def __init__(self):
        super().__init__()
        ch = (64, 128, 256, 512)

        # modality encoders
        self.rgb_backbone   = ResNet18(in_ch=3)
        self.depth_backbone = ResNet18(in_ch=1)

        # CAB modules per stage
        self.cab = nn.ModuleList([CrossModalAttention(c) for c in ch])

        # fusion backbone
        self.fusion = FusionBackbone(ch=ch)

        # head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_in = ch[-1]
        self.fc = nn.Linear(self.head_in, 1)

    def encoder(self, rgb: torch.Tensor, depth: torch.Tensor):
        R_list = self.rgb_backbone(rgb)    # [R1..R4]
        D_list = self.depth_backbone(depth)

        # CAB at each stage -> C_i
        C_list = [cab(R_list[i], D_list[i]) for i, cab in enumerate(self.cab)]

        # Fusion backbone
        F4 = self.fusion(C_list)

        # Global pooling
        g = self.gap(F4).flatten(1)                      # [B, 512]
        return g

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        g = self.encoder(rgb, depth)
        return self.fc(g)


class RGBDResNetLateFusion(nn.Module):
    def __init__(
        self,
        rgb_backbone: nn.Module | None = None,
        depth_backbone: nn.Module | None = None,
        feat_dim: int = 512,          # channels of C5 in your ResNet18
        hidden_dim: int = 256,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.rgb_backbone   = rgb_backbone   if rgb_backbone   is not None else ResNet18(in_ch=3)
        self.depth_backbone = depth_backbone if depth_backbone is not None else ResNet18(in_ch=1)
        self.head_in = feat_dim * 2

        self.gap = nn.AdaptiveAvgPool2d(1)  # GAP on C5
        self.head = nn.Sequential(
            nn.Linear(self.head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def encoder(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        c2r, c3r, c4r, c5r = self.rgb_backbone(rgb)
        c2d, c3d, c4d, c5d = self.depth_backbone(depth)

        frgb = self.gap(c5r).flatten(1)   # (B, feat_dim)
        fdep = self.gap(c5d).flatten(1)   # (B, feat_dim)

        fused = torch.cat([frgb, fdep], dim=1)  # (B, 2*feat_dim)
        return fused

    def forward(self, rgb, depth):
        feat = self.encoder(rgb, depth)
        pred = self.head(feat)
        return pred

class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # first conv can downsample via stride=2 when starting a new stage
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    def __init__(self,
                 blocks_per_stage=(2, 2, 2, 2),
                 out_ch_per_stage=(64, 128, 256, 512),
                 in_ch=3):
        super().__init__()
        # input: [B, 3, H, W]
        self.conv1 = nn.Conv2d(in_ch, out_ch_per_stage[0], kernel_size=7, stride=2, padding=3, bias=False) # [B, 64, H/2, W/2]
        self.bn1 = nn.BatchNorm2d(out_ch_per_stage[0])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # [B, 64, 160, 120]

        self.layer1 = self._make_stage(out_ch_per_stage[0], out_ch_per_stage[0], blocks_per_stage[0])            # C2 [B, 64, H/4, W/4]
        self.layer2 = self._make_stage(out_ch_per_stage[0], out_ch_per_stage[1], blocks_per_stage[1], stride=2)  # C3 [B, 128, H/8, W/8]
        self.layer3 = self._make_stage(out_ch_per_stage[1], out_ch_per_stage[2], blocks_per_stage[2], stride=2)  # C4 [B, 256, H/16, W/16]
        self.layer4 = self._make_stage(out_ch_per_stage[2], out_ch_per_stage[3], blocks_per_stage[3], stride=2)  # C5 [B, 512, H/32, W/32]

    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int,  stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResnetBlock(in_ch, out_ch, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResnetBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class RGBResNetCalories(nn.Module):
    """
    Wraps a ResNet18 backbone and adds GAP+MLP for calorie regression.
    """
    def __init__(self,
                 backbone: nn.Module | None = None,
                 in_features: int = 512, # channels from last stage (c5)
                 hidden_dim: int = 256,
                 dropout_p: float = 0.2):
        super().__init__()
        self.backbone = backbone if backbone is not None else ResNet18()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_in = in_features
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def encoder(self, x: torch.Tensor):
        c2, c3, c4, c5 = self.backbone(x)
        features = self.gap(c5).flatten(1)  # (B, in_features)
        return features

    def forward(self, rgb):
        feat = self.encoder(rgb)
        pred = self.head(feat)
        return pred

class FPN(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], out_ch=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            lateral_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            output_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, inputs):
        # inputs is a list of feature maps from backbone [c2, c3, c4, c5]
        bottom_up_features = [lateral_conv(f) for f, lateral_conv in zip(inputs, self.lateral_convs)]

        # build top-down pathway
        fpn_features = []
        x = bottom_up_features[-1]
        fpn_features.append(self.output_convs[-1](x))

        # range from second last to first input layers
        for i in range(len(bottom_up_features) - 2, -1, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest") + bottom_up_features[i] # upsample and add
            fpn_features.insert(0, self.output_convs[i](x))

        # [p2, p3, p4, p5] in dimensions of 256 channels
        # each with corresponding spatial sizes [B, C, H/2^k, W/2^k]
        return fpn_features


class SelfAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        d_k = in_ch // 2   # hidden dimension for Q,K,V

        # Q, K ,V projection
        self.W_q = nn.Conv2d(in_ch, d_k, kernel_size=1)
        self.W_k = nn.Conv2d(in_ch, d_k, kernel_size=1)
        self.W_v = nn.Conv2d(in_ch, d_k, kernel_size=1)

        # output projection
        self.W_o = nn.Conv2d(d_k, in_ch, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        d_k = C // 2
        N = H * W

        # Compute Q, K, V projections
        Q = self.W_q(x).view(B, d_k, N)          # [B, d_k, N]
        K = self.W_k(x).view(B, d_k, N)          # [B, d_k, N]
        V = self.W_v(x).view(B, d_k, N)          # [B, d_k, N]

        # Compute attention scores
        Q = Q.permute(0, 2, 1)                   # [B, N, d_k]
        attn_scores = Q @ K / (d_k ** 0.5)
        attn_weights = self.softmax(attn_scores) # [B, N, N]

        out = V @ attn_weights.permute(0, 2, 1)  # [B, d_k, N]
        out = out.view(B, d_k, H, W)
        out = self.W_o(out)

        return x + out

class BFP(nn.Module):

    def __init__(self, in_channels=256, refine_level=2):
        super().__init__()
        self.refine_level = refine_level
        self.attention = SelfAttention(in_channels)
        self.post_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        # resize all to same spatial size as refine_level (p4)
        target_size = inputs[self.refine_level].shape[2:]
        resized = [F.interpolate(f, size=target_size, mode='nearest') for f in inputs]
        # average to get balanced semantic map
        balanced = torch.mean(torch.stack(resized, dim=0), dim=0)

        refined = self.attention(balanced)
        refined = self.relu(self.bn(self.post_conv(refined)))

        # propagate back to each level
        outs = []
        for f in inputs:
            out = F.interpolate(refined, size=f.shape[2:], mode='nearest') + f
            outs.append(out)
        return outs

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

        self.channel_gate = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_gate = nn.Sigmoid()  # gate over spatial map

    def forward(self, x):
        # Input: [B, C, H, W]
        # Channel Attention
        avg_descriptor = self.global_avg_pool(x)  # [B, C, 1, 1]
        max_descriptor = self.global_max_pool(x)  # [B, C, 1, 1]

        avg_weight = self.channel_mlp(avg_descriptor)
        max_weight = self.channel_mlp(max_descriptor)
        channel_attention = self.channel_gate(avg_weight + max_weight)  # [B, C, 1, 1]

        x = x * channel_attention  # Apply channel weights, [B, C, H, W]

        # Spatial Attention
        spatial_avg = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        spatial_descriptor = torch.cat([spatial_avg, spatial_max], dim=1)  # [B, 2, H, W]
        spatial_attention = self.spatial_gate(self.spatial_conv(spatial_descriptor))  # [B, 1, H, W]

        x = x * spatial_attention  # [B, C, H, W]

        return x  # [B, C, H, W]


class RGBDFusionNet(nn.Module):
    def __init__(
        self,
        backbone_cls=ResNet18,
        fpn_out_ch: int = 256,
        refine_level: int = 2,
        spatial_dropout: float = 0.0,  # one ratio used after FPN fusion, after BFP, and after CBAM
        head_dropout: float = 0.30,    # dropout in the MLP head
    ):
        super().__init__()

        # Backbones (RGB: 3ch, Depth-raw: 1ch)
        self.rgb_backbone       = backbone_cls(in_ch=3)
        self.depth_raw_backbone = backbone_cls(in_ch=1)

        # FPNs
        in_list = [64, 128, 256, 512]
        self.rgb_fpn       = FPN(in_list, fpn_out_ch)
        self.depth_raw_fpn = FPN(in_list, fpn_out_ch)

        # BFP + CBAM
        self.bfp  = BFP(in_channels=fpn_out_ch, refine_level=refine_level)
        self.cbam = CBAM(channels=fpn_out_ch, reduction=16)

        # Single spatial dropout used at all three spots
        self.drop2d = nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity()

        self.head_in = fpn_out_ch * 4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.head_in, 256),
            nn.ReLU(),
            nn.Dropout(p=head_dropout) if head_dropout > 0 else nn.Identity(),
            nn.Linear(256, 1),
        )

    def encoder(self, rgb, depth_raw):
        # Backbones
        rgb_Cs = self.rgb_backbone(rgb)                # [C2,C3,C4,C5]
        raw_Cs = self.depth_raw_backbone(depth_raw)

        # FPNs
        rgb_Ps = self.rgb_fpn(rgb_Cs)                  # [P2..P5]
        raw_Ps = self.depth_raw_fpn(raw_Cs)

        # Fuse
        fused_Ps = [self.drop2d(r + d) for r, d in zip(rgb_Ps, raw_Ps)]

        # BFP
        refined_Ps = self.bfp(fused_Ps)
        refined_Ps = [self.drop2d(p) for p in refined_Ps]

        # CBAM
        cbam_Ps = [self.drop2d(self.cbam(p)) for p in refined_Ps]

        pooled = [self.avgpool(p) for p in cbam_Ps]    # each [B, fpn_out, 1, 1]
        fused  = torch.cat(pooled, dim=1)              # [B, 4*fpn_out, 1, 1]
        fused  = fused.flatten(1)                      # [B, 4*fpn_out]

        return fused

    def forward(self, rgb, depth_raw):
        z = self.encoder(rgb, depth_raw)
        return self.head(z)


class RGBDCaloryPredictor(nn.Module):
    def __init__(self, backbone: nn.Module, device: torch.device, arch_name: str,
                 in_features: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.device = device
        self.arch_name = arch_name
        self.bc_head = BinaryClassifierHead(in_features, hidden, p)
        self.value_head = RegressionHead(in_features, hidden, p)

    def encoder(self, batch):
        if self.arch_name in ["RGBDFusionNet", "RGBDResNetLateFusion", "RGBDFusionCAB"]:
            return self.backbone.encoder(batch["rgb"], batch["depth"])
        else:
            return self.backbone.encoder(batch["rgb"])

    def predict_zero_colary(self, batch):
        feats = self.encoder(batch)
        return self.bc_head(feats)  # logits

    def predict_colary_value(self, batch):
        feats = self.encoder(batch)
        return self.value_head(feats)

    def predict(self, batch, threshold: float = 0.5):
        feats = self.encoder(batch)
        # logits = self.bc_head(feats)
        # probs = torch.sigmoid(logits)
        #
        # # Boolean mask: True if predicted zero-calorie
        # zero_mask = probs.squeeze() < threshold

        # Predict regression output
        reg_output = self.value_head(feats)
        #reg_output = torch.expm1(reg_output)  # reverse log(1 + y) if needed

        # Combine predictions
        # preds = torch.zeros_like(reg_output)
        # preds[~zero_mask] = reg_output[~zero_mask]
        # return preds
        return reg_output

    def forward(self, batch):
        feats = self.encoder(batch)
        return self.bc_head(feats), self.value_head(feats)


class BinaryClassifierHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    def forward(self, features):  # (B,D) -> (B,1)
        return self.net(features)  # logits


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        #self.net = nn.Sequential(
        #    nn.Linear(in_features, hidden),
        #    nn.ReLU(True),
        #    nn.Dropout(p),
        #    nn.Linear(hidden, 1)
        #)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, features):  # (B,D) -> (B,1)
        #return self.net(features)
        return self.fc(features)


