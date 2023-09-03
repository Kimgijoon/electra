import math
from typing import Dict, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ModelCheckpoint, CallbackList

from src.model import Electra
import src.optimization as optimization


class Classifier(object):
    
    def __init__(self,
                 configs: Dict[str, Union[int, float]],
                 batch_size: int=None,
                 epochs: int=None,
                 lr: float=None,
                 num_gpus: Optional[int]=None,
                 ckpt_path: Optional[str]=None,
                 is_validation: Optional[bool]=None,
                 is_training: bool=False):
        """Initializer

        Args:
            configs (Dict[str, Union[int, float]]): Configuration of model hyperparameters
            batch_size (int): Number of samples per gradient update
            epochs (int): Number of epochs to train the model
            lr (float): Learning rate
            ckpt_path (Optional[str], optional): Path of checkpoint. Defaults to None.
            is_validation (Optional[bool], optional): dd. Defaults to None.
            is_training (bool): boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing).
        """
        self.configs = configs
        
        # If you have multiple gpu,
        if num_gpus:
            gpus = [f'/gpu:{x}' for x in range(num_gpus)]
            self.strategy = tf.distribute.MirroredStrategy(gpus)
        
            self.epochs = epochs
            self.batch_size = batch_size * self.strategy.num_replicas_in_sync
            self.is_validation = is_validation
            self.is_training = is_training
            
            # model, optimizer, and checkpoint must be created under `strategy.scope`.
            with self.strategy.scope():
                self.model = Electra(configs)
                trainset_size = self.configs['train']
                self.steps_per_epoch = math.ceil(trainset_size / self.batch_size)
                num_train_steps = self.steps_per_epoch * self.epochs
                num_warmup_steps = int(0.1 * num_train_steps)
                self.optimizer = optimization.create_optimizer(init_lr=lr,
                                                            num_train_steps=num_train_steps,
                                                            num_warmup_steps=num_warmup_steps,
                                                            optimizer_type='adamw')
                
                self.cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                self.train_acc = tf.keras.metrics.CategoricalAccuracy()

                if self.is_validation:
                    self.val_loss = tf.keras.metrics.Mean(name='val_loss')                
                    self.val_acc = tf.keras.metrics.CategoricalAccuracy()
                    
                    cp_callback = ModelCheckpoint(filepath=ckpt_path+'/weights-{epoch:02d}-{val_loss:.4f}.ckpt',
                                                monitor='val_loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)
                else:
                    cp_callback = ModelCheckpoint(filepath=ckpt_path+'/weights-{epoch:02d}-{loss:.4f}.ckpt',
                                                monitor='loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)
                
                # CallbackList for write log in Tensorboard
                self.callback_list = CallbackList([cp_callback], add_history=True, model=self.model)
            
        if not is_training:
            self.model = Electra(configs)
            latest = tf.train.latest_checkpoint(ckpt_path)
            self.model.load_weights(latest).expect_partial()

    def _compute_loss(self,
                      y_true: tf.Tensor,
                      y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate electra model's loss by using categorical cross entropy

        Args:
            y_true (tf.Tensor): Ground truth (correct) labels
            y_pred (tf.Tensor): Predicted labels, as returned by a model
            
        Returns:
            tf.Tensor: Loss
        """        
        loss = self.cce(y_true, y_pred)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)
        return loss
    
    def _compiled_train_step(self,
                             features: Dict[str, tf.data.Dataset],
                             labels: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor, tf.GradientTape]:
        """This is a function for compiling the model with xla (train set)

        Args:
            features (Dict[str, tf.data.Dataset]): Features dictionary in train set
            labels (tf.data.Dataset): Ground truth
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.GradientTape]: Loss and Logit, Gradient Loss and Logit obtained after model training through train set
        """    
        with tf.GradientTape() as tape:
            logits = self.model(features, self.is_training)
            loss = self._compute_loss(labels, logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return loss, logits, gradients
    
    def _compiled_val_step(self,
                           features: Dict[str, tf.data.Dataset],
                           labels: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
        """This is a function for compiling the model with xla (val set)

        Args:
            features (Dict[str, tf.data.Dataset]): Features dictionary in validation set
            labels (tf.data.Dataset): Ground truth
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Loss and Logit obtained after model validation through validation set
        """    
        logits = self.model(features, False)
        loss = self._compute_loss(labels, logits)
        return loss, logits
    
    def _train_step(self,
                    inputs: Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset],
                    use_xla: Optional[bool]=None) -> tf.Tensor:
        """Custom training loop

        Args:
            inputs (Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset]): Features dictionary and ground trouth
            use_xla (Optional[bool], optional): Flags that determine whether to use XLA. Defaults to None.

        Returns:
            tf.Tensor: loss, logit, gradient
        """
        features, labels = inputs
        
        compile_step = tf.function(self._compiled_train_step, jit_compile=use_xla)
        loss, logits, gradients = compile_step(features, labels)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_acc.update_state(logits, labels)
        return loss
    
    def _val_step(self,
                  inputs: Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset],
                  use_xla: Optional[bool]=None):
        """Custom validating loop.

        Args:
            inputs (Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset]): Features dictionary and ground trouth
            use_xla (Optional[bool], optional): Flags that determine whether to use XLA. Defaults to None.
        """
        features, labels = inputs
        
        compile_step = tf.function(self._compiled_val_step, jit_compile=use_xla)
        loss, logits = compile_step(features, labels)
        self.val_loss.update_state(loss)
        self.val_acc.update_state(logits, labels)
    
    @tf.function
    def distributed_train_step(self,
                               inputs: Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset],
                               use_xla: Optional[bool]=None) -> tf.Tensor:
        """This is a function for performing model training with multi-gpu (train set)

        Args:
            inputs (Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset]): Features dictionary and ground trouth
            use_xla (Optional[bool], optional): Flags that determine whether to use XLA. Defaults to None.
            
        Returns:
            tf.Tensor: train loss
        """
        per_replica_losses = self.strategy.run(self._train_step, args=(inputs, use_xla,))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_losses,
                                    axis=None)
        return loss
    
    @tf.function
    def distributed_val_step(self,
                             inputs: Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset],
                             use_xla: Optional[bool]=None):
        """This is a function for performing model training with multi-gpu (val set)

        Args:
            inputs (Tuple[Dict[str, tf.data.Dataset], tf.data.Dataset]): Features dictionary and ground trouth
            use_xla (Optional[bool], optional): Flag that determine whether to use XLA. Defaults to None.
        """        
        self.strategy.run(self._val_step, args=(inputs, use_xla,))
    
    def train(self,
              train_set: tf.data.Dataset,
              val_set: Optional[tf.data.Dataset]=None,
              summaries_path: Optional[str]=None,
              use_xla: Optional[bool]=None):
        """_summary_

        Args:
            train_set (tf.data.Dataset): Train dataset
            val_set (Optional[tf.data.Dataset], optional): Validation Dataset. Defaults to None.
            summaries_path (Optional[str], optional): Path of tensorboard's logs. Defaults to None
            use_xla (Optional[[bool], optional): Flags that determine whether to use XLA. Defaults to None.
            
        Ref:
            XLA(Accelerated Linear Algebra): https://www.tensorflow.org/xla?hl=ko
            Class-balanced loss: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        """
        metrics_names = ['train_loss', 'train_acc', 'val_loss']
        
        # convert tf.data.Dataset to tf.distribute tf.distribute.DistributedDataset
        # bc use multi gpu
        train_set: tf.distribute.DistributedDataset = self.strategy.experimental_distribute_dataset(train_set)
        if self.is_validation:
            val_set: tf.distribute.DistributedDataset = self.strategy.experimental_distribute_dataset(val_set)
        
        # log info to tensorboard
        train_summary_writer = tf.summary.create_file_writer(f'{summaries_path}/train')
        test_summary_writer = tf.summary.create_file_writer(f'{summaries_path}/valid')
        
        # train start
        # logs = {}
        # self.callback_list.on_train_begin(logs=logs)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            
            progbar = Progbar(self.steps_per_epoch * self.batch_size, stateful_metrics=metrics_names)
            for train_step, train_inputs in enumerate(train_set):
                # self.callback_list.on_batch_begin(train_step, logs=logs)
                # self.callback_list.on_train_batch_begin(train_step, logs=logs)
                
                # compute train loss and metric
                train_loss = self.distributed_train_step(train_inputs, use_xla=use_xla)
                
                # logs['train_loss'] = train_loss
                # self.callback_list.on_train_batch_end(train_step, logs=logs)
                # self.callback_list.on_batch_end(train_step, logs=logs)
                
                values = [('train_loss', train_loss), ('train_acc', self.train_acc.result())]
                progbar.update((train_step+1)*self.batch_size, values=values)
            
            if self.is_validation:
                for val_step, val_inputs in enumerate(val_set):
                    # self.callback_list.on_batch_begin(val_step, logs=logs)
                    # self.callback_list.on_test_batch_begin(val_step, logs=logs)
                    
                    # compute validation loss and metric
                    self.distributed_val_step(val_inputs, use_xla=use_xla)
                    
                    # self.callback_list.on_test_batch_end(val_step, logs=logs)
                    # self.callback_list.on_batch_end(val_step, logs=logs)
                    
                    values = [
                        ('train_loss', train_loss),
                        ('train_acc', self.train_acc.result()),
                        ('val_loss', self.val_loss.result()),
                        ('val_acc', self.val_acc.result())]
                progbar.update((train_step+1)*self.batch_size, values=values, finalize=True)                
                # logs['val_loss'] = self.val_loss.result()
            
                # write valid log    
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                    tf.summary.scalar('acc', self.val_acc.result(), step=epoch)
            
            # write train log
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('acc', self.train_acc.result(), step=epoch)
                tf.summary.scalar('lr', self.optimizer._decayed_lr('float32').numpy(), step=epoch)
            
            self.train_acc.reset_states()
            if self.is_validation:
                self.val_acc.reset_states()
                self.val_loss.reset_states()
            # self.callback_list.on_epoch_end(epoch, logs=logs)

        # train end
        # self.callback_list.on_train_end(logs=logs)
    
    def predict(self, problem: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict type of input problem. This will return probability of classes for given data.

        Args:
            problem (str): Input problem
            ckpt_dir (str): Path of checkpoint

        Returns:
            Tuple[np.ndarray, np.ndarray]: The probability per each class and inferenced result
        """
        problem = {x: tf.constant([problem[x]]) for x in problem}
        prob, _ = self.model(problem).numpy().squeeze()
        result = np.argmax(prob)
        return prob, result

    def get_embedding_vector(self, problem: str) -> np.ndarray:
        """Returns the model’s input embeddings.

        Args:
            problem (str): Input problem

        Returns:
            np.ndarray: Embedding vector about given data(Problem)
        """        
        problem = {x: tf.constant([problem[x]]) for x in problem}
        
        # pooled output
        _, emb_vec = self.model(problem).numpy().squeeze()
        return emb_vec
