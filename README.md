## Machine Learning

A repository of common machine learning algorithms and example usages.

🧠 1. Supervised Learning Models
a. Regression

    What: Predict continuous values.

    Example: Predict house prices, accident count, temperature, or stock prices.

b. Classification

    What: Predict class labels.

    Example: Spam detection, disease diagnosis, image classification (cat vs. dog).

🧩 2. Unsupervised Learning Models
a. Clustering

    What: Group similar data points.

    Example: Customer segmentation, traffic accident hotspots.

b. Dimensionality Reduction

    What: Reduce number of features.

    Example: Visualizing high-dimensional data, preprocessing for modeling.

🔁 3. Sequence Models
a. Recurrent Neural Networks (RNN)

    What: Handle sequential/time-series data.

    Example: Predict weather patterns, accident trends, stock market.

b. Long Short-Term Memory (LSTM) / GRU

    What: Improved RNNs with memory control.

    Example: Text generation, traffic prediction, music generation.

🔍 4. Attention-Based Models
a. Transformer

    What: State-of-the-art for sequence tasks.

    Example: Translation, summarization, time-series forecasting.

b. BERT / GPT-style Models

    What: Pretrained language understanding/generation.

    Example: Chatbots, document search, summarization.

🧱 5. Convolutional Neural Networks (CNN)

    What: Process grid-like data such as images or time-frequency.

    Example: Object detection, medical imaging, self-driving car vision.

📦 6. Autoencoders

    What: Compress and reconstruct data.

    Example: Anomaly detection (e.g., rare accident types), denoising, latent representation learning.

💡 7. Generative Models
a. Generative Adversarial Networks (GANs)

    What: Generate new, realistic data.

    Example: Synthetic images, data augmentation, video generation.

b. Variational Autoencoders (VAEs)

    What: Probabilistic generation of data.

    Example: Face morphing, anomaly detection.

🤖 8. Reinforcement Learning Models

    What: Learn via trial and error with rewards.

    Example: Traffic signal optimization, autonomous driving policies, dynamic pricing.

📈 9. Time Series Forecasting Models

    What: Predict future values.

    Example: Sales prediction, accident trends, resource demand.

🔐 10. Recommendation Systems

    What: Suggest relevant items.

    Example: Movie/music recommendations, driver insurance personalization.

🌐 11. Graph Neural Networks (GNN)

    What: Model relationships between entities.

    Example: Social network analysis, traffic network optimization.

🧬 12. Multi-Modal Models

    What: Combine text, image, audio, etc.

    Example: Video captioning, accident report analysis with images + text.

🧠 13. Meta Learning / Few-Shot Learning

    What: Learn from few examples.

    Example: Medical imaging (rare diseases), fraud detection.

🛠️ 14. Custom Models with Functional or Subclassing APIs

    What: Full control over model behavior.

    Example: Building hybrid architectures (e.g., CNN+LSTM), interpretable AI.

🧮 15. Probabilistic Models (with TensorFlow Probability)

    What: Add uncertainty awareness.

    Example: Bayesian neural networks for risk assessment, probabilistic forecasting.


# TensorFlow Model Templates by Category


## 1. Regression (Supervised Learning)
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 2. Classification
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## 3. Clustering (using KMeans from Scikit-learn)
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(data)
```

## 4. Dimensionality Reduction (PCA with sklearn)
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

## 5. RNN (for sequence data)
```python
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(timesteps, features)),
    tf.keras.layers.SimpleRNN(64),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 6. LSTM
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 7. Transformer (Simple version)
```python
input_layer = tf.keras.layers.Input(shape=(seq_len, d_model))
attention = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)(input_layer, input_layer)
output = tf.keras.layers.Dense(1)(attention)
model = tf.keras.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
```

## 8. BERT (via Hugging Face)
```python
from transformers import TFBertModel
bert = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
outputs = bert(input_ids, attention_mask=attention_mask)[0][:, 0, :]
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
```

## 9. CNN
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## 10. Autoencoder
```python
input_img = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(64, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = tf.keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

## 11. GAN
```python
# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(784, activation='sigmoid')
])
# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN
z = tf.keras.Input(shape=(100,))
img = generator(z)
validity = discriminator(img)

gan = tf.keras.Model(z, validity)
discriminator.trainable = False
gan.compile(optimizer='adam', loss='binary_crossentropy')
```

## 12. VAE
```python
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
z_mean = tf.keras.layers.Dense(32)(x)
z_log_var = tf.keras.layers.Dense(32)(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z])
```

## 13. Reinforcement Learning (with TF-Agents)
```python
# Install tf-agents
# pip install tf-agents
from tf_agents.environments import suite_gym
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common

env = suite_gym.load('CartPole-v0')
q_net = q_network.QNetwork(env.observation_spec(), env.action_spec())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
train_step = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step
)
agent.initialize()
```

## 14. Time Series Forecasting (Using LSTM)
```python
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(time_steps, features)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 15. Recommendation Systems (Matrix Factorization)
```python
user_input = tf.keras.layers.Input(shape=(1,))
item_input = tf.keras.layers.Input(shape=(1,))
user_embed = tf.keras.layers.Embedding(num_users, 50)(user_input)
item_embed = tf.keras.layers.Embedding(num_items, 50)(item_input)

dot_product = tf.keras.layers.Dot(axes=2)([user_embed, item_embed])
model = tf.keras.Model([user_input, item_input], dot_product)
model.compile(optimizer='adam', loss='mse')
```

## 16. Graph Neural Networks (with Spektral)
```python
# pip install spektral
from spektral.layers import GCNConv
X_in = tf.keras.Input(shape=(num_features,))
A_in = tf.keras.Input((None,), sparse=True)
x = GCNConv(32, activation='relu')([X_in, A_in])
x = GCNConv(1)([x, A_in])
model = tf.keras.Model(inputs=[X_in, A_in], outputs=x)
model.compile(optimizer='adam', loss='mse')
```

## 17. Multi-Modal Model (Text + Image)
```python
text_input = tf.keras.Input(shape=(text_len,))
img_input = tf.keras.Input(shape=(height, width, channels))

text_branch = tf.keras.layers.Embedding(10000, 64)(text_input)
text_branch = tf.keras.layers.GlobalAveragePooling1D()(text_branch)

img_branch = tf.keras.applications.ResNet50(include_top=False, pooling='avg')(img_input)

combined = tf.keras.layers.concatenate([text_branch, img_branch])
output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
model = tf.keras.Model(inputs=[text_input, img_input], outputs=output)
```

## 18. Few-Shot Learning (Siamese Network)
```python
def create_base_network(input_shape):
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(input, x)

input_a = tf.keras.Input(shape=(input_dim,))
input_b = tf.keras.Input(shape=(input_dim,))

base_network = create_base_network((input_dim,))
processed_a = base_network(input_a)
processed_b = base_network(input_b)

L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([processed_a, processed_b])
prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L1_distance)
model = tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

## 19. Probabilistic Model (TensorFlow Probability)
```python
import tensorflow_probability as tfp
tfp = tfp.layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tfp.DenseFlipout(1)
])
model.compile(optimizer='adam', loss='mse')
```

## 20. Custom Model with Functional API
```python
inputs = tf.keras.Input(shape=(input_dim,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
```
