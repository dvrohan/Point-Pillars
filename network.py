import tensorflow as tf

class PFNLayer:
    def __init__(self, last_layer, in_channels, out_channels, use_norm=True):
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            self.norm = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", epsilon=1e-3, momentum=0.99)
            self.linear = tf.keras.layers.Dense(self.units, input_shape=(in_channels,), activation=None, use_bias=False)

class PointPillars:
    def __init__(self, num_filters=(64,), use_norm=True):
        num_input_features = 9
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = tf.keras.Sequential(pfn_layers)

