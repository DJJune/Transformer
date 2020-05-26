from tqdm import tqdm
import argparse
from symbols import _vocab_to_id, _id_to_vocab
import hyperparameters as hp
from network import Transformer
import torch as t
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


def load_checkpoint(step, model_name="transformer"):
    state_dict = t.load(hp.checkpoint_path +
                        '/checkpoint_%s_%d.pth.tar' % (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text_list, args):
    """
    Attention !!!
    During training procedure, we can feed the whole right-shifted target sentence at once
    However, during testing, the model has to output the word step by step

       a      b    c  [eos]
       ^      ^    ^    ^
       |      |    |    |
    [start]   a    b    c
    """

    # restore model
    m = Transformer()
    m.load_state_dict(load_checkpoint(args.restore, "transformer"))

    for text in text_list:
        print("[INPUT] " + text)
        source_seq = []
        for w in text.split(" "):
            source_seq.append(_vocab_to_id[w])
        source_seq.append(_vocab_to_id["-"])   # add eos

        source_tensor = t.LongTensor(
            np.asarray(source_seq)).unsqueeze(0).cuda()

        decoder_input = t.zeros([1, 1]).long().cuda()

        pos_source = t.arange(1, source_tensor.size(1) +
                              1).long().unsqueeze(0).cuda()

        m = m.cuda()
        m.eval()

        pbar = range(args.max_len)
        with t.no_grad():
            for i in pbar:
                pos_target = t.arange(
                    1, decoder_input.size(1)+1).unsqueeze(0).cuda()
                pred, attn, attn_enc, attn_dec = m.forward(
                    source_tensor, decoder_input, pos_source, pos_target)

                # get the latest word from decoder output
                output_word_idx = t.topk(
                    F.softmax(pred[:, -1, :], dim=1), k=1)[1]

                if output_word_idx.squeeze().cpu() == 1:
                    # output eos
                    break

                # put it at the end of the decoder input and the predict next word
                decoder_input = t.cat([decoder_input, output_word_idx], dim=1)

        output_sentence = []
        for w in decoder_input.squeeze().cpu().numpy()[1:]:
            output_sentence.append(_id_to_vocab[w])
        print("[OUTPUT] " + " ".join(output_sentence))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=int,
                        help='Global step to restore checkpoint', default=100)
    parser.add_argument('--max_len', type=int,
                        help='max length of output', default=10)
    args = parser.parse_args()

    text_list = ["how are you", "what is your name"]
    synthesis(text_list, args)
