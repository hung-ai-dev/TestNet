Pretrained model:
- Efficient noisy student
    - tf_efficientnet_b0_ns
    - tf_efficientnet_b1_ns
    - tf_efficientnet_b2_ns
    - tf_efficientnet_b3_ns
    - tf_efficientnet_b4_ns
    - tf_efficientnet_b5_ns
    - tf_efficientnet_b6_ns
    - tf_efficientnet_b7_ns

- Efficient net
    - efficientnet_b0
    - efficientnet_b1
    - efficientnet_b2
    - efficientnet_b3
    - efficientnet_b4
    - efficientnet_b5
    - efficientnet_b6
    - efficientnet_b7

Specify model: __model = tf_efficientnet_b4_ns(True)__
Specify data path:
- train_set = ProductImageLoader(__image_folder__, __train_csv_file__, 'train')
- val_set = ProductImageLoader(__image_folder__, __val_csv_file__, 'val')