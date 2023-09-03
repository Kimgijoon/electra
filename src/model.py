from typing import Dict, Union

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import TruncatedNormal
from transformers.models.electra.modeling_tf_electra import TFElectraModel, TFElectraPooler


class Electra(tf.keras.Model):
    
    def __init__(self, configs: Dict[str, Union[int, float]]):
        """Initialize each layers

        Args:
            configs (Dict[str, Union[int, float]]): Configuration for Electra model
        """        
        super(Electra, self).__init__()
        self.config = configs
        
        self.electra_layer = TFElectraModel.from_pretrained('monologg/koelectra-base-v3-discriminator',
                                                            from_pt=True)
        # self.pooler = TFElectraPooler(self.electra_layer.config)
        self.dropout = Dropout(configs['dropout_prob'])
        self.output_layer = Dense(configs['class_num'],
                                  kernel_initializer=TruncatedNormal(self.electra_layer.config.initializer_range), 
                                  activation='softmax')

    def call(self, features: Dict[str, tf.Tensor], training: bool) -> tf.Tensor:
        """A Electra model on huggingface.
        The architecture is based on the paper "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators".

        Args:
            features (Dict[str, tf.Tensor]): tf.data.Dataset
            training (bool): Python boolean indicating whether the layer should behave in
                                training mode (adding dropout) or in inference mode

        Returns:
            tf.Tensor: Logit
        """
        sequence_output = self.electra_layer(features['input_ids'],
                                             attention_mask=features['attention_mask'],
                                             token_type_ids=features['token_type_ids'])[0]
        cls_token = sequence_output[:, 0, :]    # take <s> token (equiv. to [CLS])
        logit = self.dropout(cls_token, training=training)
        logit = self.output_layer(logit)
        
        return logit
