import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import re

batch_size = 100
train_steps = 1000

train_df = pd.read_csv('../data/Mental_Health_Clean.csv')

train_df.columns = map(
    lambda x: x.replace(" ", "_").replace("?", "").replace("(", "").replace(")", "").replace(",", "").replace(":", ""),
    train_df.columns)
print(train_df.columns)

output_col = 'Have_you_been_diagnosed_with_a_mental_health_condition_by_a_medical_professional'
feature_cols = train_df.columns.tolist()
feature_cols.remove(output_col)
to_remove = [
    'Have_you_had_a_mental_health_disorder_in_the_past',
    'Do_you_currently_have_a_mental_health_disorder',
    'Have_you_ever_sought_treatment_for_a_mental_health_issue_from_a_mental_health_professional',
    'If_you_have_a_mental_health_issue_do_you_feel_that_it_interferes_with_your_work_when_being_treated_effectively',
    'If_you_have_a_mental_health_issue_do_you_feel_that_it_interferes_with_your_work_when_NOT_being_treated_effectively'
]
for item in to_remove:
    feature_cols.remove(item)

X = train_df[feature_cols]
y = train_df[output_col]

feature_columns = list(map(tf.feature_column.numeric_column, feature_cols))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

model = tf.compat.v1.estimator.DNNClassifier(feature_columns=feature_columns,
                                    hidden_units=[16],
                                    optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
                                      learning_rate=0.1,
                                      l1_regularization_strength=0.001
                                    ))

model.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size), steps=train_steps)

# Evaluate the model.
eval_result = model.evaluate(
    input_fn=lambda:eval_input_fn(X_test, y_test, batch_size))

print('\nTest set accuracy: {accuracy:0.2f}\n'.format(**eval_result))
accuracy = eval_result['accuracy'] * 100
