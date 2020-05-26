
# Export all vocab:

# define you vocab here, the followed one is just an example
_pad = '~'
_eos = '-'
vocab = ['how', 'are', 'you', 'what', 'is', 'your',
         'name', 'I', 'am', 'fine', ',', 'thanks', 'Bob']

# _pad and _eos must be placed at the first and second position
# because we pad 0 and at 1 at the the of sequence
vocab = [_pad, _eos] + vocab

# Mappings from vocab to numeric ID and vice versa:
_vocab_to_id = {s: i for i, s in enumerate(vocab)}
_id_to_vocab = {i: s for i, s in enumerate(vocab)}

if __name__ == '__main__':
    print(vocab)
