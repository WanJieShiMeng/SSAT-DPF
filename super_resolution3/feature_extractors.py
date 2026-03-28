import sys
import torch
from torch import nn
from typing import List
import torch.nn.functional as F

device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else


def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        # print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    elif model_type == 'mae':
        # print("Creating MAE Feature Extractor...")
        feature_extractor = FeatureExtractorMAE(**kwargs)
    elif model_type == 'swav':
        # print("Creating SwAV Feature Extractor...")
        feature_extractor = FeatureExtractorSwAV(**kwargs)
    elif model_type == 'swav_w2':
        # print("Creating SwAVw2 Feature Extractor...")
        feature_extractor = FeatureExtractorSwAVw2(**kwargs)
    elif model_type == 'ae':
        # print("Creating SwAVw2 Feature Extractor...")
        feature_extractor = FeatureExtractorAE(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor


def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        '''
        Parent feature extractor class.

        param: model_path: path to the pretrained model
        param: input_activations:
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        # print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):
    '''
    Wrapper to extract features from pretrained DDPMs.

    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''

    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps

        # Save decoder activations
        for idx, block in enumerate(self.model.output_blocks):
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)

    def _load_pretrained_model(self, model_path, **kwargs):
        from diffusion_model3.unet import UNetModelConnect
        from diffusion_model3.diffusion import Diffusion
        self.model = UNetModelConnect(
                           image_size=32,
                           in_channels=kwargs['spe'],
                           model_channels=128,
                           out_channels=kwargs['spe'],
                           num_res_blocks=2,
                           attention_resolutions=set([4,8]),
                           num_heads=4,
                           num_heads_upsample=-1,
                           num_head_channels=-1,
                           channel_mult=(1,2,3,4),
                           dropout=0.0,
                           conv_resample=True,
                           dims=2,
                           num_classes=None,
                           use_checkpoint=False,
                           use_fp16=False,
                           use_scale_shift_norm=False,
                           resblock_updown=False,
                           use_new_attention_order=False,
                        ).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.diffusion = Diffusion(T=1000)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x):
        activations = []
        for t in self.steps:
            # Compute x_t and run DDPM
            ti = torch.full((1,), t, dtype=torch.long).to(device)
            noisy_x,_ = self.diffusion.forward_diffusion_sample(x, ti, device=device)
            self.model(noisy_x, ti)

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # cat在一起
        size = (activations[-1].shape[2], activations[-1].shape[3])
        fea_all_t = []
        for fea in activations:
            fea = F.interpolate(fea, size=size,mode='bilinear')
            if len(fea_all_t) == 0:
                fea_all_t = fea
            else:
                fea_all_t = torch.cat([fea_all_t, fea], 1)
        # Per-layer list of activations [N, C, H, W]
        return fea_all_t


class FeatureExtractorMAE(FeatureExtractor):
    '''
    Wrapper to extract features from pretrained MAE
    '''

    def __init__(self, num_blocks=12, **kwargs):
        super().__init__(**kwargs)
        # Save features from deep encoder blocks
        for layer in self.model.blocks[-num_blocks:]:
            layer.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer)

    def _load_pretrained_model(self, model_path, **kwargs):
        import mae
        from functools import partial
        sys.path.append(mae.__path__[0])
        from mae.models_mae import MaskedAutoencoderViT

        # Create MAE with ViT-L-8 backbone
        model = MaskedAutoencoderViT(
            img_size=32, patch_size=4, in_chans=102, embed_dim=64, depth=5, num_heads=4,
            decoder_embed_dim=64, decoder_depth=5, decoder_num_heads=4,
            mlp_ratio=0.125, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False)

        model.load_state_dict(torch.load(model_path))
        self.model = model.eval().to(device)

    @torch.no_grad()
    def forward(self, x, **kwargs):
        _, _, ids_restore = self.model.forward_encoder(x, mask_ratio=0)
        ids_restore = ids_restore.unsqueeze(-1)
        sqrt_num_patches = int(self.model.patch_embed.num_patches ** 0.5)
        activations = []
        for block in self.feature_blocks:
            # remove cls token
            a = block.activations[:, 1:]
            # unshuffle patches
            a = torch.gather(a, dim=1, index=ids_restore.repeat(1, 1, a.shape[2]))
            # reshape to obtain spatial feature maps
            a = a.permute(0, 2, 1)
            a = a.view(*a.shape[:2], sqrt_num_patches, sqrt_num_patches)
            activations.append(a)
            block.activations = None
        # Per-layer list of activations [N, C, H, W]
        size = (activations[-1].shape[2], activations[-1].shape[3])
        fea_all_t = []
        for fea in activations:
            fea = F.interpolate(fea, size=size, mode='bilinear')
            if len(fea_all_t) == 0:
                fea_all_t = fea
            else:
                fea_all_t = torch.cat([fea_all_t, fea], 1)
        return fea_all_t


class FeatureExtractorAE(FeatureExtractor):
    '''
    Wrapper to extract features from pretrained SwAVs
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layer = self.model.decoder
        layer.register_forward_hook(self.save_hook)
        self.feature_blocks.append(layer)

    def _load_pretrained_model(self, model_path, **kwargs):
        from MTGAN.autoencoder import AE
        model = AE(kwargs["spe"]).to(device)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, x, **kwargs):
        self.model(x)
        activations = []
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations[0]


class FeatureExtractorSwAV(FeatureExtractor):
    '''
    Wrapper to extract features from pretrained SwAVs
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layers = [self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]

        # Save features from sublayers
        for layer in layers:
            for l in layer[::2]:
                l.register_forward_hook(self.save_hook)
                self.feature_blocks.append(l)

    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50

        model = resnet50(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        self.model = model.module.eval()

    @torch.no_grad()
    def forward(self, x, **kwargs):
        self.model(x)

        activations = []
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorSwAVw2(FeatureExtractorSwAV):
    '''
    Wrapper to extract features from twice wider pretrained SwAVs
    '''

    def _load_pretrained_model(self, model_path, **kwargs):
        import swav
        sys.path.append(swav.__path__[0])
        from swav.hubconf import resnet50w2

        model = resnet50w2(pretrained=False).to(device).eval()
        model.fc = nn.Identity()
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        self.model = model.module.eval()


def collect_features(activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = (activations[-1].shape[2], activations[-1].shape[3])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode='bilinear'
        )
        resized_activations.append(feats[0])

    return torch.cat(resized_activations, dim=0)

if __name__ == "__main__":
    k = {
        "model_type":'ddpm',
        "blocks":[0,1,2,3,4,5,6,7,8,9,10,11],
        "input_activations": True,
        "steps":[10],
        "model_path":"/home/wxy/diffusion_super_resolution/diffusion_model3/save_model/Unet_T=500/PaviaC_p=32/unet.pkl",
        'spe':102,
    }
    extractor = create_feature_extractor(**k)
    x = torch.randn(102,32,32)
    feature = extractor(x)
    print(feature.shape)
    print(type(feature))
    print(feature.device)
    for i in range(len(feature)):
        print(feature[i].shape)
    cat_fea = collect_features(activations=feature)
    print(cat_fea.shape)
