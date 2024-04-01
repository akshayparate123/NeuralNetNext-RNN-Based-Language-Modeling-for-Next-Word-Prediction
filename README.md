# NeuralNetNext-RNN-Based-Language-Modeling-for-Next-Word-Prediction
NeuralNetNext: RNN-Based Language Modeling for Next Word Prediction

#### Name: Akshay Parate


N-Gram

```python
tokenizer = Tokenizer()
tokenizer.fit(train_sentences[:2])
print()

token_occurrence = sorted(tokenizer.token_occurrence.items(), key=lambda e: e[1], reverse=True)
for token, occurrence in token_occurrence[:10]:
    print(token, ':', occurrence)
print()
sent_token_ids = tokenizer.encode(train_sentences[:2])
print()
for original_sentence, token_ids in zip(train_sentences[:2], sent_token_ids):
    sentence = [tokenizer.inverse_vocab[token] for token in token_ids]
    print(original_sentence, sentence, '\n')
```

    Fitting Tokenizer: 2 / 2
    The number of distinct tokens: 44
    
    n : 2
    aer : 1
    banknote : 1
    berlitz : 1
    calloway : 1
    centrust : 1
    cluett : 1
    fromstein : 1
    gitano : 1
    guterman : 1
    
    Encoding with Tokenizer: 2 / 2
    
     aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts rake regatta rubens sim snack-food ssangyong swapo wachter 
     ['<s>', 'aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano', 'guterman', 'hydro', 'quebec', 'ipo', 'kia', 'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens', 'sim', 'snack', 'food', 'ssangyong', 'swapo', 'wachter', '</s>'] 
    
     pierre <unk> N years old will join the board as a nonexecutive director nov. N 
     ['<s>', 'pierre', 'n', 'years', 'old', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'nov', 'n', '</s>'] 
    
    

    

### Calculate unigram and bigram count


```python
bigram = BiGram(unigram_count, bigram_count)

# Perplexity
bigram_perplexity = perplexity(bigram, valid_token_ids)
print(f'The perplexity of Bigram is: {bigram_perplexity:.4f}')
```

    Calculating perplexity: 3352 / 3352
    
    The perplexity of Bigram is: 325.8354
    


```python
gt = GoodTuring(unigram_count, bigram_count, threshold=100)

# Perplexity
gt_perplexity = perplexity(gt, valid_token_ids)
print(f'The perplexity of Good Turing is: {gt_perplexity:.4f}')
```

    Calculating perplexity: 3352 / 3352
    
    The perplexity of Good Turing is: 130.5334
    


```python
kn = KneserNey(unigram_count, bigram_count, d=0.75)

# Perplexity
kn_perplexity = perplexity(kn, valid_token_ids)
print(f'The perplexity of Kneser-Ney is: {kn_perplexity:.4f}')
```

    Calculating perplexity: 3352 / 3352
    
    The perplexity of Kneser-Ney is: 62.5908
    

### Predict the next word given a previous word


```python
def predict(model: 'BiGram', w1: int, vocab_size: int):
    """ Predict the w2 with the hightest probability given w1

    Args:
        model: A BiGram, GoodTuring, or KneserNey model that has the calc_prob function
        w1: current word
        vocab_size: the number of tokens in the vocabulary
    """
    result = None
    highest_prob = 0
    for w2 in range(1, vocab_size):
        # Start your code here
        prob = model.calc_prob(w1, w2)
        if prob > highest_prob:
            result = w2
            highest_prob = prob
        # End
    return result
```

Bigram next word prediction


```python
np.random.seed(12345)

vocab_size = len(tokenizer)
indexes = np.random.choice(len(test_token_ids), 10, replace=False)
for i in indexes:
    token_ids = test_token_ids[i][1:-1]
    print(' '.join([tokenizer.inverse_vocab[token_id] for token_id in token_ids]) + ' ____')
    pred = predict(bigram, token_ids[-1], vocab_size)
    print(f'predicted last token: {tokenizer.inverse_vocab[pred]}')
    print('---------------------------------------------')
```

    sharply falling stock prices do reduce consumer wealth damage business ____
    predicted last token: </s>
    ---------------------------------------------
    but robert an official of the association said no ____
    predicted last token: longer
    ---------------------------------------------
    it also has interests in military electronics and marine ____
    predicted last token: s
    ---------------------------------------------
    first chicago since n has reduced its loans to such ____
    predicted last token: as
    ---------------------------------------------
    david m jones vice president at g ____
    predicted last token: s
    ---------------------------------------------
    the n stock specialist firms on the big board floor ____
    predicted last token: traders
    ---------------------------------------------
    at the same time the business was hurt by ____
    predicted last token: the
    ---------------------------------------------
    salomon will cover the warrants by buying sufficient shares or ____
    predicted last token: n
    ---------------------------------------------
    in july southmark corp the dallas based real estate and financial ____
    predicted last token: services
    ---------------------------------------------
    he concluded his remarks by and at some ____
    predicted last token: of
    ---------------------------------------------
    

Good Turing next word prediction


```python
np.random.seed(12345)

vocab_size = len(tokenizer)
indexes = np.random.choice(len(test_token_ids), 10, replace=False)
for i in indexes:
    token_ids = test_token_ids[i][1:-1]
    print(' '.join([tokenizer.inverse_vocab[token_id] for token_id in token_ids]) + ' ____')
    pred = predict(gt, token_ids[-1], vocab_size)
    print(f'predicted last token: {tokenizer.inverse_vocab[pred]}')
    print('---------------------------------------------')
```

    sharply falling stock prices do reduce consumer wealth damage business ____
    predicted last token: </s>
    ---------------------------------------------
    but robert an official of the association said no ____
    predicted last token: longer
    ---------------------------------------------
    it also has interests in military electronics and marine ____
    predicted last token: s
    ---------------------------------------------
    first chicago since n has reduced its loans to such ____
    predicted last token: as
    ---------------------------------------------
    david m jones vice president at g ____
    predicted last token: s
    ---------------------------------------------
    the n stock specialist firms on the big board floor ____
    predicted last token: traders
    ---------------------------------------------
    at the same time the business was hurt by ____
    predicted last token: the
    ---------------------------------------------
    salomon will cover the warrants by buying sufficient shares or ____
    predicted last token: n
    ---------------------------------------------
    in july southmark corp the dallas based real estate and financial ____
    predicted last token: officer
    ---------------------------------------------
    he concluded his remarks by and at some ____
    predicted last token: of
    ---------------------------------------------
    

Kneser-Ney next word prediction


```python
np.random.seed(12345)

vocab_size = len(tokenizer)
indexes = np.random.choice(len(test_token_ids), 10, replace=False)
for i in indexes:
    token_ids = test_token_ids[i][1:-1]
    print(' '.join([tokenizer.inverse_vocab[token_id] for token_id in token_ids]) + ' ____')
    pred = predict(kn, token_ids[-1], vocab_size)
    print(f'predicted last token: {tokenizer.inverse_vocab[pred]}')
    print('---------------------------------------------')
```

    sharply falling stock prices do reduce consumer wealth damage business ____
    predicted last token: </s>
    ---------------------------------------------
    but robert an official of the association said no ____
    predicted last token: </s>
    ---------------------------------------------
    it also has interests in military electronics and marine ____
    predicted last token: </s>
    ---------------------------------------------
    first chicago since n has reduced its loans to such ____
    predicted last token: as
    ---------------------------------------------
    david m jones vice president at g ____
    predicted last token: </s>
    ---------------------------------------------
    the n stock specialist firms on the big board floor ____
    predicted last token: </s>
    ---------------------------------------------
    at the same time the business was hurt by ____
    predicted last token: the
    ---------------------------------------------
    salomon will cover the warrants by buying sufficient shares or ____
    predicted last token: n
    ---------------------------------------------
    in july southmark corp the dallas based real estate and financial ____
    predicted last token: </s>
    ---------------------------------------------
    he concluded his remarks by and at some ____
    predicted last token: of
    ---------------------------------------------
    

## RNN 

```python
from tensorflow.keras import Model

class RNN(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        """ Init of the RNN model

        Args:
            vocab_size, embedding_dim: used for initialze the embedding layer.
            hidden_units: number of hidden units of the RNN layer.
        """
        super().__init__()
        # Start your code here
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(hidden_units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        # End

    def call(self, x):
        """ Forward of the RNN model

        Args:
            x: Tensor, (batch_size x max_seq_len). Input tokens. Here, max_seq_len is the longest length of sentences in this batch becasue we did pad_batch.
        Return:
            outputs: Tensor, (batch_size x max_seq_len x vocab_size). Logits for every time step. !!!NO SOFTMAX HERE!!!
        """
        # Start your code here
        embedded = self.embedding(x)  # (batch_size, max_seq_len, embedding_dim)
        rnn_output = self.rnn(embedded)  # (batch_size, max_seq_len, hidden_units)
        outputs = self.dense(rnn_output)
        # End
        return outputs
```

### RNN language model

```python
vocab_size = len(tokenizer)
hidden_units = 128
embedding_dim = 64
num_epoch = 30
batch_size = 256
```


```python
model = RNN(vocab_size, embedding_dim, hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```

### Train RNN


```python
num_samples = len(x_train)
n_batch = int(np.ceil(num_samples / batch_size))
n_valid_batch = int(np.ceil(len(x_valid) / batch_size))
for epoch in range(num_epoch):
    epoch_loss = 0.0
    for batch_idx in range(n_batch):
        start = batch_idx * batch_size
        end = start + batch_size
        x_batch, y_batch, seq_lens_batch = x_train[start:end], y_train[start:end], train_seq_lens[start:end]
        real_batch_size = len(x_batch)
        x_batch, y_batch, seq_lens_batch = pad_batch(x_batch, y_batch, seq_lens_batch, pad_val=tokenizer.pad_token_id)
        with tf.GradientTape() as tape:
            output = model(x_batch)
            loss = seq2seq_loss(output, y_batch, seq_lens_batch)

        if batch_idx % 1 == 0 or batch_idx == num_samples - 1:
            print_line(f'Epoch {epoch + 1} / {num_epoch} - Step {batch_idx + 1} / {n_batch} - loss: {loss:.4f}')

        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        epoch_loss += loss * real_batch_size

    valid_loss = 0.0
    for batch_idx in range(n_valid_batch):
        start = batch_idx * batch_size
        end = start + batch_size
        x_batch, y_batch, seq_lens_batch = x_valid[start:end], y_valid[start:end], valid_seq_lens[start:end]
        real_batch_size = len(x_batch)
        x_batch, y_batch, seq_lens_batch = pad_batch(x_batch, y_batch, seq_lens_batch, pad_val=tokenizer.pad_token_id)
        output = model(x_batch)
        loss = seq2seq_loss(output, y_batch, seq_lens_batch)

        if batch_idx % 1 == 0 or batch_idx == len(x_valid) - 1:
            print_line(f'Epoch {epoch + 1} / {num_epoch} - Step {batch_idx + 1} / {n_valid_batch} - loss: {loss:.4f}')

        valid_loss += loss * real_batch_size
    print(f'\rEpoch {epoch + 1} / {num_epoch} - Step {n_batch} / {n_batch} - train loss: {epoch_loss / num_samples:.4f} - valid loss: {valid_loss / len(x_valid):.4f}')
```

    Epoch 1 / 30 - Step 170 / 170 - train loss: 6.9871 - valid loss: 6.6481
    Epoch 2 / 30 - Step 170 / 170 - train loss: 6.6823 - valid loss: 6.6370
    Epoch 3 / 30 - Step 170 / 170 - train loss: 6.6368 - valid loss: 6.4843
    Epoch 4 / 30 - Step 170 / 170 - train loss: 6.3856 - valid loss: 6.2237
    Epoch 5 / 30 - Step 170 / 170 - train loss: 6.1472 - valid loss: 6.0347
    Epoch 6 / 30 - Step 170 / 170 - train loss: 5.9497 - valid loss: 5.8651
    Epoch 7 / 30 - Step 170 / 170 - train loss: 5.7855 - valid loss: 5.7475
    Epoch 8 / 30 - Step 170 / 170 - train loss: 5.6618 - valid loss: 5.6647
    Epoch 9 / 30 - Step 170 / 170 - train loss: 5.5633 - valid loss: 5.6003
    Epoch 10 / 30 - Step 170 / 170 - train loss: 5.4787 - valid loss: 5.5414
    Epoch 11 / 30 - Step 170 / 170 - train loss: 5.4026 - valid loss: 5.4929
    Epoch 12 / 30 - Step 170 / 170 - train loss: 5.3344 - valid loss: 5.4511
    Epoch 13 / 30 - Step 170 / 170 - train loss: 5.2728 - valid loss: 5.4181
    Epoch 14 / 30 - Step 170 / 170 - train loss: 5.2166 - valid loss: 5.3863
    Epoch 15 / 30 - Step 170 / 170 - train loss: 5.1650 - valid loss: 5.3633
    Epoch 16 / 30 - Step 170 / 170 - train loss: 5.1180 - valid loss: 5.3489
    Epoch 17 / 30 - Step 170 / 170 - train loss: 5.0749 - valid loss: 5.3340
    Epoch 18 / 30 - Step 170 / 170 - train loss: 5.0362 - valid loss: 5.3161
    Epoch 19 / 30 - Step 170 / 170 - train loss: 4.9979 - valid loss: 5.2981
    Epoch 20 / 30 - Step 170 / 170 - train loss: 4.9614 - valid loss: 5.2831
    Epoch 21 / 30 - Step 170 / 170 - train loss: 4.9266 - valid loss: 5.2718
    Epoch 22 / 30 - Step 170 / 170 - train loss: 4.8944 - valid loss: 5.2631
    Epoch 23 / 30 - Step 170 / 170 - train loss: 4.8636 - valid loss: 5.2566
    Epoch 24 / 30 - Step 170 / 170 - train loss: 4.8352 - valid loss: 5.2515
    Epoch 25 / 30 - Step 170 / 170 - train loss: 4.8084 - valid loss: 5.2487
    Epoch 26 / 30 - Step 170 / 170 - train loss: 4.7825 - valid loss: 5.2427
    Epoch 27 / 30 - Step 170 / 170 - train loss: 4.7579 - valid loss: 5.2377
    Epoch 28 / 30 - Step 170 / 170 - train loss: 4.7344 - valid loss: 5.2365
    Epoch 29 / 30 - Step 170 / 170 - train loss: 4.7144 - valid loss: 5.2415
    Epoch 30 / 30 - Step 170 / 170 - train loss: 4.6963 - valid loss: 5.2420
    

### Perplexity of RNN

Here,
1. you need to calculate the perplexity based on its definition.
2. Besides, you need to record the loss for every word prediction and calculate the sum of loss
3. Finaly, you will need to compare the perplexity by definition and the perplexity by the loss: `np.exp(total_loss / n_words)`

```python
n = len(x_valid)
log_probs = 0
n_words = 0  # number of words to predict in the entire dataset
total_loss = 0  # total loss of each word's loss
for i in range(n):
# for i in range(1):

    if i % 1 == 0 or i == n - 1:
        print_line('Calculating perplexity:', (i + 1), '/', n)
    x_line, y_line, line_seq_lens = x_valid[i:i + 1], y_valid[i: i + 1], valid_seq_lens[i:i + 1]
    x_line, y_line, line_seq_lens = pad_batch(x_line, y_line, line_seq_lens, tokenizer.pad_token_id)
    output = model(x_line)
    pred_probs = tf.nn.softmax(output, axis=-1)

    for real_token, probs in zip(y_line[0], pred_probs[0]):
        # Start your code here
        log_probs += np.log2(probs[real_token])
        n_words = n_words+1
        # End
    loss = 0
    # Start your code here
    loss = seq2seq_loss(output,y_line,line_seq_lens)
    total_loss = total_loss + loss * len(x_line[0])
    # End
print('\n')
# print(n_words)
# print(log_probs)
perplexity = 2 ** ((-1 / n_words) * log_probs)
print(f'Perplexity by definition: {perplexity:.4f}, Perplexity by loss: {np.exp(total_loss / n_words):.4f}')

# If you implement correctly, the two perplexity will be almost the same.
```

    Calculating perplexity: 3352 / 3352
    
    Perplexity by definition: 188.8956, Perplexity by loss: 188.8954
    

### Predict the next word given a previous sentence


```python
np.random.seed(12345)

vocab_size = len(tokenizer)
indexes = np.random.choice(len(test_token_ids), 10, replace=False)
for i in indexes:
    token_ids = test_token_ids[i][1:-1]
    print(' '.join([tokenizer.inverse_vocab[token_id] for token_id in token_ids]) + ' ____')
    x = tf.convert_to_tensor(token_ids, dtype=tf.int64)  # now x is a tensor of (seq_len, )
    # Start your code here
    x = np.reshape(x,(1,-1))
    prob = model.call(x)
    output = prob[0]
    n_t = output[-1]
    pred = np.argmax(n_t)
    # End
    print(f'predicted last token: {tokenizer.inverse_vocab[pred]}')
    print('---------------------------------------------')
```

    sharply falling stock prices do reduce consumer wealth damage business ____
    predicted last token: and
    ---------------------------------------------
    but robert an official of the association said no ____
    predicted last token: longer
    ---------------------------------------------
    it also has interests in military electronics and marine ____
    predicted last token: </s>
    ---------------------------------------------
    first chicago since n has reduced its loans to such ____
    predicted last token: as
    ---------------------------------------------
    david m jones vice president at g ____
    predicted last token: s
    ---------------------------------------------
    the n stock specialist firms on the big board floor ____
    predicted last token: traders
    ---------------------------------------------
    at the same time the business was hurt by ____
    predicted last token: the
    ---------------------------------------------
    salomon will cover the warrants by buying sufficient shares or ____
    predicted last token: by
    ---------------------------------------------
    in july southmark corp the dallas based real estate and financial ____
    predicted last token: services
    ---------------------------------------------
    he concluded his remarks by and at some ____
    predicted last token: of
    ---------------------------------------------
    

## Conclusion

Briefly analyze the result of N-Gram and RNN

### N-gram:

1. **Counting Word Frequencies**: Initially, the text is divided into single words (unigrams) and pairs of consecutive words (bigrams). These counts are pivotal for the N-gram model to predict subsequent words based on the frequency of these word pairs.

2. **Utilizing Bigram Model**: The model employs bigrams to anticipate the next word by considering the frequency of occurrence of word pairs within the training data.

3. **Application of Smoothing Techniques**: Techniques such as Good Turing and Kneser-Ney smoothing are applied to refine the probability estimates of N-gram models, enhancing the accuracy of predictions.

4. **Perplexity Evaluation**: Perplexity, which gauges how well a probability distribution anticipates a given sample, is computed for each model. Among these, the Kneser-Ney smoothing model exhibits the lowest perplexity, trailed by Good Turing, and finally, the Bigram model.

### RNN:

1. **Model Structure**: The architecture comprises an embedding layer to transform words into compact vectors, a hidden SimpleRNN layer for capturing sequential dependencies, and an output layer for making predictions.

2. **Addressing Variable Length Sequences**: To facilitate batch processing in the RNN model, padding is applied to ensure that sentences are of uniform length.

3. **Loss Metric Calculation**: The sequence-to-sequence loss is determined by averaging the loss across both axes for every sample in the batch.

4. **Performance Assessment**: Post-training, the validation loss and training loss are assessed. Additionally, perplexity, calculated using both conventional methods and total loss, is provided to evaluate the RNN model's performance.

### Comparison:

1. **Complexity and Training Time**: Unlike the RNN model, the N-gram model is less intricate and quicker to train due to its non-iterative nature.

2. **Similarity in Predictions**: Despite their structural disparities, both models yield comparable predictions for subsequent words, suggesting they capture analogous patterns in the data.

3. **Perplexity Analysis**: While the RNN model boasts a lower perplexity compared to the Bigram model, it falls short of the levels achieved by Good Turing and Kneser-Ney smoothing models. This implies that while RNNs offer potential for capturing nuanced patterns, N-gram models with smoothing techniques provide more accurate predictions.

4. **Balancing Complexity and Nuance**: Although the RNN model offers complexity and potential for capturing intricate patterns, it requires greater computational resources and time for training. Hence, the choice between the models hinges on the specific task requirements and the desired level of prediction accuracy.


```python

```
