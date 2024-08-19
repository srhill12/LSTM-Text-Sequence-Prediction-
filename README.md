
# Text Sequence Prediction using LSTM

This project implements a sequence prediction model using Long Short-Term Memory (LSTM) neural networks. The goal is to train a model that can predict the next word in a sequence based on a corpus of text.

## Project Overview

The project is designed to accomplish the following tasks:
1. **Text Preprocessing**: Read in text, clean it, and tokenize it.
2. **Sequence Generation**: Create sequences of tokens to be used as training data.
3. **Model Development**: Develop an LSTM model to predict the next word in a sequence.
4. **Model Training**: Train the LSTM model using the prepared sequences.
5. **Evaluation**: Evaluate the model's performance during training.

## Setup Instructions

### Prerequisites

Ensure that you have Python installed along with the following libraries:

- `spacy`
- `tensorflow`
- `keras`
- `numpy`

You can install these libraries using pip:

```bash
pip install spacy tensorflow keras numpy
```

Additionally, download the large English language model for SpaCy:

```bash
python -m spacy download en_core_web_lg
```

### Running the Project

1. **Import Required Libraries**

    Start by importing the required libraries and loading the SpaCy model with disabled components:

    ```python
    import spacy
    nlp = spacy.load('en_core_web_lg', disable=["tagger", "ner", "lemmatizer"])
    ```

2. **Read and Preprocess the Text**

    The `read_file` function reads in a text file and saves the content as a string. The `separate_punc` function cleans the text by removing punctuation and converting it to lowercase.

    ```python
    def read_file(filepath):
        with open(filepath) as f:
            str_text = f.read()
        return str_text

    def separate_punc(holmes_text):
        return [token.text.lower() for token in nlp(holmes_text) \
                if token.text not in '\n\n \n\n\n!"“”-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n']

    # Example usage
    holmes_text = read_file('Resources/A_Case_Of_Identity.txt')
    tokens = separate_punc(holmes_text)
    ```

3. **Generate Sequences**

    Generate sequences of tokens where each sequence contains 25 words to predict the 26th word:

    ```python
    train_len = 26
    text_sequences = []

    for i in range(train_len, len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)
    ```

4. **Tokenization and Conversion to Arrays**

    Convert the sequences of tokens into numeric arrays using Keras' `Tokenizer`:

    ```python
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)

    sequences = tokenizer.texts_to_sequences(text_sequences)
    num_sequences = np.array(sequences)
    ```

5. **Create Input Sequences and One-Hot Encode the Target**

    Prepare the data for training by splitting the sequences into input (`X`) and target (`y`) and then one-hot encode the target:

    ```python
    X = num_sequences[:,:-1]
    y = num_sequences[:,-1]
    y = to_categorical(y, num_classes=vocabulary_size+1)
    ```

6. **LSTM Model Creation**

    Create and compile the LSTM model:

    ```python
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Embedding

    def create_model(vocabulary_size, seq_len):
        model = Sequential()
        model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
        model.add(LSTM(150, return_sequences=True))
        model.add(LSTM(150))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(vocabulary_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    ```

7. **Model Training**

    Train the model using the prepared data:

    ```python
    model = create_model(vocabulary_size+1, seq_len)
    model.fit(X, y, epochs=300, batch_size=128, verbose=1)
    ```

## Conclusion

This project demonstrates the process of using an LSTM model to predict the next word in a sequence based on the previous words. It covers text preprocessing, sequence generation, model development, and training.
```

Feel free to modify the `README.md` content to suit your specific needs or add more details as necessary.
