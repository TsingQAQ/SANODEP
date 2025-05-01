from trieste.models.interfaces import TrainablePredictJointReparamModelStack
from trieste.types import TensorType
from trieste.data import Dataset
import tensorflow as tf


class TrainablePredictJointPredictReparamModelStack(TrainablePredictJointReparamModelStack):
    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        means, vars_ = zip(*[model.conditional_predict_joint(query_points,\
                                                              Dataset(additional_data.query_points, additional_data.observations[..., idx:idx+1])) for idx, model in enumerate(self._models)])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

    def conditional_predict_f_sample(self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        samples = [model.conditional_predict_f_sample(query_points,\
                                                              Dataset(additional_data.query_points, additional_data.observations[..., idx:idx+1]), num_samples) for idx, model in enumerate(self._models)]
        return tf.concat(samples, axis=-1)
    
    def conditional_predict_f(self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        means, vars_ = zip(*[model.conditional_predict_f(query_points,\
                                                              Dataset(additional_data.query_points, additional_data.observations[..., idx:idx+1])) for idx, model in enumerate(self._models)])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

