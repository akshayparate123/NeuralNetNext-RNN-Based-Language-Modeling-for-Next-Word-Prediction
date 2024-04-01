# NeuralNetNext-RNN-Based-Language-Modeling-for-Next-Word-Prediction
NeuralNetNext: RNN-Based Language Modeling for Next Word Prediction

#### Name: Akshay Parate


N-Gram


```python
import sys


def print_line(*args):
    """ Inline print and go to the begining of line
    """
    args1 = [str(arg) for arg in args]
    str_ = ' '.join(args1)
    print('\r' + str_, end='')
```


```python
import tensorflow as tf


# If you are going to use GPU, make sure the GPU is in the output
tf.config.list_physical_devices('GPU')
```




    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]




```python
from typing import List, Tuple, Union, Dict

import numpy as np
```

### Load Data & Preprocessing



```python
import os
import pickle


data_path = 'a3-data'

train_sentences = open(os.path.join(data_path, 'train.txt')).readlines()
valid_sentences = open(os.path.join(data_path, 'valid.txt')).readlines()
test_sentences = open(os.path.join(data_path, 'input.txt')).readlines()
print('number of train sentences:', len(train_sentences))
print('number of valid sentences:', len(valid_sentences))
print('number of test sentences:', len(test_sentences))
```

    number of train sentences: 42068
    number of valid sentences: 3370
    number of test sentences: 3165
    


```python
import re


class Preprocessor:
    def __init__(self, punctuation=True, url=True, number=True):
        self.punctuation = punctuation
        self.url = url
        self.number = number

    def apply(self, sentence: str) -> str:
        """ Apply the preprocessing rules to the sentence
        Args:
            sentence: raw sentence
        Returns:
            sentence: clean sentence
        """
        sentence = sentence.lower()
        sentence = sentence.replace('<unk>', '')
        if self.url:
            sentence = Preprocessor.remove_url(sentence)
        if self.punctuation:
            sentence = Preprocessor.remove_punctuation(sentence)
        if self.number:
            sentence = Preprocessor.remove_number(sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    @staticmethod
    def remove_punctuation(sentence: str) -> str:
        """ Remove punctuations in sentence with re
        Args:
            sentence: sentence with possible punctuations
        Returns:
            sentence: sentence without punctuations
        """
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        return sentence

    @staticmethod
    def remove_url(sentence: str) -> str:
        """ Remove urls in text with re
        Args:
            sentence: sentence with possible urls
        Returns:
            sentence: sentence without urls
        """
        sentence = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', ' ', sentence)
        return sentence

    @staticmethod
    def remove_number(sentence: str) -> str:
        """ Remove numbers in sentence with re
        Args:
            sentence: sentence with possible numbers
        Returns:
            sentence: sentence without numbers
        """
        sentence = re.sub(r'\d+', ' ', sentence)
        return sentence
```


```python
class Tokenizer:
    def __init__(self, sos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>'):
        # Special tokens.
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.mask_token = mask_token

        self.vocab = { sos_token: 0, eos_token: 1, pad_token: 2, unk_token: 3, mask_token: 4 }  # token -> id
        self.inverse_vocab = { 0: sos_token, 1: eos_token, 2: pad_token, 3: unk_token, 4: mask_token }  # id -> token
        self.token_occurrence = { sos_token: 0, eos_token: 0, pad_token: 0, unk_token: 0, mask_token: 0 }  # token -> occurrence

        self.preprocessor = Preprocessor()

    @property
    def sos_token_id(self):
        """ Create a property method.
            You can use self.sos_token_id or tokenizer.sos_token_id to get the id of the sos_token.
        """
        return self.vocab[self.sos_token]

    @property
    def eos_token_id(self):
        return self.vocab[self.eos_token]

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def mask_token_id(self):
        return self.vocab[self.mask_token]

    def __len__(self):
        """ A magic method that enable program to know the number of tokens by calling:
            ```python
            tokenizer = Tokenizer()
            num_tokens = len(tokenizer)
            ```
        """
        return len(self.vocab)

    def fit(self, sentences: List[str]):
        """ Fit the tokenizer using all sentences.
        1. Tokenize the sentence by splitting with spaces.
        2. Record the occurrence of all tokens
        3. Construct the token to index (self.vocab) map and the inversed map (self.inverse_vocab) based on the occurrence. The token with a higher occurrence has the smaller index

        Args:
            sentences: All sentences in the dataset.
        """
        n = len(sentences)
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == n - 1:
                print_line('Fitting Tokenizer:', (i + 1), '/', n)
            tokens = self.preprocessor.apply(sentence.strip()).split()
            if len(tokens) <= 1:
                continue
            for token in tokens:
                if token == '<unk>':
                    continue
                self.token_occurrence[token] = self.token_occurrence.get(token, 0) + 1
        print_line('\n')

        token_occurrence = sorted(self.token_occurrence.items(), key=lambda e: e[1], reverse=True)
        for token, occurrence in token_occurrence[:-5]:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token

        print('The number of distinct tokens:', len(self.vocab))

    def encode(self, sentences: List[str]) -> List[List[int]]:
        """ Encode the sentences into token ids
            Note: 1. if a token in a sentence does not exist in the fit encoder, we ignore it.
                  2. If the number of tokens in a sentence is less than two, we ignore this sentence.
                  3. Note that, for every sentence, we will add an sos_token, i.e., the id of <s> at the start of the sentence,
                     and add an eos_token, i.e., the id of </s> at the end of the sentence.
        Args:
            sentences: Raw sentences
        Returns:
            sent_token_ids: A list of id list
        """
        n = len(sentences)
        sent_token_ids = []
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 or i == n - 1:
                print_line('Encoding with Tokenizer:', (i + 1), '/', n)
            token_ids = []
            tokens = self.preprocessor.apply(sentence.strip()).split()
            for token in tokens:
                if token == '<unk>':
                    continue
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
            if len(token_ids) <= 1:
                continue
            token_ids = [self.sos_token_id] + token_ids + [self.eos_token_id]
            sent_token_ids.append(token_ids)
        print_line('\n')
        return sent_token_ids
```


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
    
    


```python
tokenizer = Tokenizer()
tokenizer.fit(train_sentences)
train_token_ids = tokenizer.encode(train_sentences)
valid_token_ids = tokenizer.encode(valid_sentences)
test_token_ids = tokenizer.encode(test_sentences)
```

    Fitting Tokenizer: 42068 / 42068
    The number of distinct tokens: 9614
    Encoding with Tokenizer: 42068 / 42068
    Encoding with Tokenizer: 3370 / 3370
    Encoding with Tokenizer: 3165 / 3165
    

### Calculate unigram and bigram count


```python
def get_unigram_count(train_token_ids: List[List[int]]) -> Dict:
    """ Calculate the occurrence of each token in the dataset.

    Args:
        train_token_ids: each element is a list of token ids
    Return:
        unigram_count: A map from token_id to occurrence
    """
    unigram_count = {}
    # Start your code here
    for token_id in train_token_ids:
        for token in token_id:
            if token in unigram_count:
                unigram_count[token] = unigram_count[token] + 1
            else:
                unigram_count[token] = 1
    # End
    return unigram_count
```


```python
def get_bigram_count(train_token_ids: List[List[int]]) -> Dict[int, Dict]:
    """ Calculate the occurrence of bigrams in the dataset.

    Args:
        train_token_ids: each element is a list of token ids
    Return:
        bigram_count: A map from token_id to next token occurrence. Key: token_id, value: Dict[token_id -> occurrence]
                      For example, {
                          5: { 10: 5, 20: 4 }
                      } means (5, 10) occurs 10 times, (5, 20) occurs 4 times.
    """
    bigram_count = {}
    # Start your code here
    for token_id in train_token_ids:
        for i in range(0,len(token_id)-1):

            if token_id[i] in bigram_count:
                # print("If {} in {}".format(token_id[i],bigram_count))

                if token_id[i+1] in bigram_count[token_id[i]]:
                    # print("If {} in {}".format(token_id[i+1],bigram_count[token_id[i]]))
                    bigram_count[token_id[i]][token_id[i+1]] = bigram_count[token_id[i]][token_id[i+1]] + 1
                    # print("Increment {}".format(bigram_count[token_id[i]][token_id[i+1]]))
                else:
                    # print("If {} not in {}".format(token_id[i+1],bigram_count[token_id[i]]))
                    bigram_count[token_id[i]][token_id[i+1]] = 1
                    # print("Initialize {} as 1".format(bigram_count[token_id[i]][token_id[i+1]]))
            else:
                # print("If {} not in {}".format(token_id[i],bigram_count))
                bigram_count[token_id[i]] = {token_id[i+1]:1}
                # print("Initialize {} as 1".format(bigram_count))
    # End
    return bigram_count
```


```python
unigram_count = get_unigram_count(train_token_ids)
bigram_count = get_bigram_count(train_token_ids)
# bigram_count = get_bigram_count([[1,2,3,4,5,6,7],[5,6,2,3,9],[3,5,9,1,6,5]])
# unigram_count = get_unigram_count([[1,2,3,4,5,6,7],[5,6,2,3,9],[3,5,9,1,6,5]])

```

### BiGram


```python
class BiGram:
    def __init__(self, unigram_count, bigram_count):
        self.unigram_count = unigram_count
        self.bigram_count = bigram_count

    def calc_prob(self, w1: int, w2: int) -> float:
        """ Calculate the probability of p(w2 | w1) using the BiGram model.

        Args:
            w1, w2: current token and next token
        Note:
            if prob you calculated is 0, you should return 1e-5.
        """
        # Start your code here
        count_w1 = 0
        count_w1_w2 = 0
        prob = 0
        if w1 in self.bigram_count:
            # print("Yes {} in {}".format(w1,bigram_count))
            if w2 in self.bigram_count[w1]:
                # print("Yes {} in {}".format(w2,bigram_count[w1]))
                count_w1_w2 = self.bigram_count[w1][w2]
                for w in list(self.bigram_count[w1].values()):
                    count_w1 = count_w1 + w
                # print(count_w1_w2,count_w1)
                prob = count_w1_w2/count_w1
            else:
                prob = 1e-5
        else:
            prob = 1e-5
        # End
        return prob
```


```python
# inc = 1
# for count in bigram_count:
#     # if inc < 0:
#     #     inc = inc + 1
#     #     continue
#     if inc > 5:
#         break
#     inc = inc + 1
#     print("{}:{}".format(count,bigram_count[count]))
#     print()
```


```python
from scipy.optimize import curve_fit


def power_law(x, a, b):
    """ Power law to fit the number of occurrence
    """
    return a * np.power(x, b)


class GoodTuring(BiGram):
    def __init__(self, unigram_count, bigram_count, threshold=100):
        super().__init__(unigram_count, bigram_count)
        self.threshold = threshold
        self.bigram_Nc = self.calc_Nc()
        self.bi_c_star, self.bi_N = self.smoothing(self.bigram_Nc)

    def calc_Nc(self) -> Dict[int, Union[float, int]]:
        """ You need to calculate Nc of bigram

        Return:
            bigram_Nc: A map from count to the occurrence (count of count)
                       For example {
                           10: 78
                       } means there are 78 bigrams occurs 10 times in the dataset.
                       Also, 10 is a small c, for large c, it's occurrence will be replaced with the power law.
        """
        bigram_Nc = {}
        # Start your code here
        # Count the occurrence of count in self.bigram_count.
        for bigram in self.bigram_count:
            for count in self.bigram_count[bigram]:
                if self.bigram_count[bigram][count] in bigram_Nc:
                    bigram_Nc[self.bigram_count[bigram][count]] = bigram_Nc[self.bigram_count[bigram][count]]+1
                else:
                    bigram_Nc[self.bigram_count[bigram][count]] = 1
        # End
        # print(bigram_Nc)
        self.replace_large_c(bigram_Nc)
        return bigram_Nc

    def replace_large_c(self, Nc):
        """ Fit with power law
        """
        x, y = zip(*sorted(Nc.items(), reverse=True))
        popt, pcov = curve_fit(power_law, x, y, bounds=([0, -np.inf], [np.inf, 0]))
        a, b = popt

        max_count = max(Nc.keys())
        for c in range(self.threshold + 1, max_count + 2):
            Nc[c] = power_law(c, a, b)

    def smoothing(self, Nc: Dict[int, Union[float, int]]) -> Tuple[Dict[int, float], float]:
        """ Calculate the c_star and N

        Args:
            self.bigram_Nc
        Returns:
            c_star: The mapping from bigram count to smoothed count
            N: The sum of c multiplied by Nc
        """
        c_star = {}
        N = 0
        max_count = max(Nc.keys())
        # print(max_count)
        # Start your code here
        for count in range(0,max_count):
            if count == 0:
                c_star[count] = Nc[1]
            else:
                c_star[count] = (count + 1) *Nc[count+1]/Nc[count]
                N += count * Nc[count]
        # End
        c_star[max_count] = max_count
        return c_star, N

    def calc_prob(self, w1, w2):
        """ Calculate the probability of p(w2 | w1) using the Good Turing model.

        Args:
            w1, w2: current token and next token
        Note:
            1. The numerator is the smoothed bigram count of (w1, w2)
            2. The denominator is the unigram count of w1
            3. You should be careful to distinguish when (w1, w2) does not exists in the training data.
        """
        prob = 0
        # Start your code here
        bigram = (w1, w2)
        unigram = w1
        # print(self.bi_c_star.keys())
        if w1 in self.bigram_count and w2 in self.bigram_count[w1]:
                numerator = self.bigram_count[w1][w2]
                numerator = self.bi_c_star[numerator]
                denominator = self.unigram_count[w1]
                # print(numerator,denominator)
                prob = numerator / denominator
        else:
                numerator = self.bigram_Nc[1]/self.bi_N
                denominator = self.unigram_count[w1]
                prob = numerator / denominator
        return prob
```


```python
# gt = GoodTuring(unigram_count, bigram_count, threshold=100)
# # # Perplexity
# gt_perplexity = perplexity(gt, valid_token_ids)
# print(f'The perplexity of Good Turing is: {gt_perplexity:.4f}')
```


```python

```

### Kneser-Ney


```python
class KneserNey(BiGram):
    def __init__(self, unigram_count, bigram_count, d=0.75):
        super().__init__(unigram_count, bigram_count)
        self.d = d

        self.lambda_ = self.calc_lambda()
        self.p_continuation = self.calc_p_continuation()

    def calc_lambda(self):
        """ Calculate the λ(w)

        Return:
            lambda_: A dict from token_id (w) to λ(w).
        """
        lambda_ = {}
        # Start your code here
        for word in self.unigram_count:
            if word in self.bigram_count:
                word2 = len(self.bigram_count[word])
            else:
                word2 = 0
            lambda_[word] = (self.d*word2)/self.unigram_count[word]
        # End
        return lambda_

    def calc_p_continuation(self):
        """ Calculate the p_continuation(w)

        Return:
            lambda_: A dict from token_id (w) to λ(w).
        """
        numerator = {}  # token -> type of previous token
        denominator = len(self.bigram_count)  # type of all previous tokens
        # Start your code here
        for w in self.unigram_count:
                c = 0
                for w1List in list(self.bigram_count.values()):
                    if w in w1List:
                        c = c+1
                numerator[w] = c
        # End
        p_continuation = { 0: 0, 2: 0, 3: 0, 4: 0 }
        for w, count in numerator.items():
            p_continuation[w] = count / denominator
        return p_continuation

    def calc_prob(self, w1, w2):
        """ Calculate the probability of p(w2 | w1) using the Kneser-Ney model.

        Args:
            w1, w2: current token and next token
        """
        # Start your code here
        c_w1_w2 = self.bigram_count[w1][w2] if w1 in self.bigram_count and w2 in self.bigram_count[w1] else 0
        prob = max(c_w1_w2 - self.d, 0) / self.unigram_count[w1] + self.lambda_[w1] * self.p_continuation[w2]
        # End
        return prob
```


### Perplexity 


```python
import math


def perplexity(model, token_ids):
    """ Calculate the perplexity score.

    Args:
        model: the model you want to evaluate (BiGram, GoodTuring, or KneserNey)
        token_ids: a list of validation token_ids
    Return:
        perplexity: the perplexity of the model on texts
    Note:

    """
    log_probs = 0
    n = len(token_ids)
    n_words = 0
    for i, tokens in enumerate(token_ids):
        if i % 100 == 0 or i == n - 1:
            print_line('Calculating perplexity:', (i + 1), '/', n)
        # Start your code here
        # Calculate log probability for each bigram in the sequence
        for j in range(len(tokens) - 1):
            w1 = tokens[j]
            w2 = tokens[j + 1]
            log_probs += math.log(model.calc_prob(w1, w2) + 1e-200)  # Add a small value to avoid log(0)
            n_words += 1

        # End

    perp = 0
    # Start your code here
    # Calculate the final perplexity
    perp = math.exp(-log_probs / n_words)
    # End
    print('\n')

    return perp
```


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

### Split feature and label


```python

```


```python
def get_feature_label(token_ids: List[List[int]], window_size: int=-1):
    """ Split features and labels for the training, validation, and test datasets.

    Note:
        If window size is -1, for a sentence with n tokens,
            it selects the tokens rangeing from [0, n - 1) as the feature,
            and selects tokens ranging from [1, n) as the label.
        Otherwise, it divides a sentence with multiple windows and do the previous split.
    """
    x = []
    y = []
    seq_lens = []
    for sent_token_ids in token_ids:
        if window_size == -1:
            x.append(sent_token_ids[:-1])
            y.append(sent_token_ids[1:])
            seq_lens.append(len(sent_token_ids) - 1)
        else:
            if len(sent_token_ids) > window_size:
                sub_sent_size = window_size + 1
                n_window = len(sent_token_ids) // (sub_sent_size)
                for i in range(n_window):
                    start = i * sub_sent_size
                    sub_sent = sent_token_ids[start:(start + sub_sent_size)]
                    x.append(sub_sent[:-1])
                    y.append(sub_sent[1:])
                    seq_lens.append(len(sub_sent) - 1)
                if len(sent_token_ids) % sub_sent_size > 0:
                    sub_sent = sent_token_ids[-sub_sent_size:]
                    x.append(sub_sent[:-1])
                    y.append(sub_sent[1:])
                    seq_lens.append(len(sub_sent) - 1)
            else:
                x.append(sent_token_ids[:-1])
                y.append(sent_token_ids[1:])
                seq_lens.append(len(sent_token_ids) - 1)
    return x, y, seq_lens
```


```python
window_size = 40
x_train, y_train, train_seq_lens = get_feature_label(train_token_ids, window_size)
x_valid, y_valid, valid_seq_lens = get_feature_label(valid_token_ids)
x_test, y_test, test_seq_lens = get_feature_label(valid_token_ids)
print(max(train_seq_lens), max(valid_seq_lens), max(test_seq_lens))
```

    40 68 68
    


```python

```

### Pad sentences in a batch to equal length


```python
def pad_batch(x_batch: List[List[int]], y_batch: List[List[int]], seq_lens_batch: List[int], pad_val: int):
    """ Pad the sentences in a batch with pad_val based on the longest sentence.

    Args:
        x_batch, y_batch, seq_lens_batch: the input data
        pad_val: the padding value you need to fill to pad the sentences to the longest sentence.

    Return:
        x_batch: Tensor, (batch_size x max_seq_len)
        y_batch: Tensor, (batch_size x max_seq_len)
        seq_lens_batch: Tensor, (batch_size, )
    """
    max_len = max(seq_lens_batch)
    # Start your code here
    num_sent = len(seq_lens_batch)

    # Padding the sentence to the length of the longest sentence
    x_batch = [x_batch[s] + [pad_val]*(max_len - len(x_batch[s])) for s in range(num_sent)]
    y_batch = [y_batch[s] + [pad_val]*(max_len - len(y_batch[s])) for s in range(num_sent)]
    # End
    x_batch, y_batch = tf.convert_to_tensor(x_batch, dtype=tf.int64), tf.convert_to_tensor(y_batch, dtype=tf.int64)
    seq_lens_batch = tf.convert_to_tensor(seq_lens_batch, dtype=tf.int64)
    return x_batch, y_batch, seq_lens_batch
```


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

```

### Seq2seq loss


```python
from tensorflow_addons.seq2seq import sequence_loss


def seq2seq_loss(logits, target, seq_lens):
    """ Calculate the sequence to sequence loss using the sequence_loss from tensorflow

    Args:
        logits: Tensor (batch_size x max_seq_len x vocab_size). The output of the RNN model.
        target: Tensor (batch_size x max_seq_len). The groud-truth of words.
        seq_lens: Tensor (batch_size, ). The real sequence length before padding.
    """
    loss = 0
    # Start your code here
    # 1. make a sequence mask (batch_size x max_seq_len) using tf.sequence_mask. This is to build a mask with 1 and 0.
    mask = tf.sequence_mask(seq_lens, maxlen=tf.shape(target)[1], dtype=tf.float32)
    #    Entry with 1 is the valid time step without padding. Entry with 0 is the time step with padding. We need to exclude this time step.
    # 2. calculate the loss with sequence_loss. Carefully read the documentation of each parameter
    loss = sequence_loss(logits, target,mask,average_across_timesteps = True,average_across_batch = True)
    # End
    return loss
```


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

```


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
