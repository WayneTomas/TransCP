dataset='unc'

output_dir='outputs/unc/public'

checkpoint_best = True
batch_size=32
epochs=100
lr_drop=60
freeze_epochs=10
freeze_modules=['backbone']
load_weights_path='pretrained_checkpoints/detr-r50-unc.pth'

model_config = dict(
    decoder=dict(
        type='VisualDenstanglingPrototype',
        num_queries=1,
        query_dim=256,
        return_intermediate=True,
        num_extra_layers=1,
        extra_layer=dict(
            type='DiscriminativeFeatEncLayer',
            d_model=256,
            img_query_with_pos=False,
            img2text_attn_args=dict(
                type='MultiheadAttention',
                embed_dim=256, num_heads=8, dropout=0.1
            ),
            discrimination_coef_settings=dict(
                text_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                img_proj=dict(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1),
                scale=1.0,
                sigma=0.5,
                pow=2.0,
            ),
        )
    )
)