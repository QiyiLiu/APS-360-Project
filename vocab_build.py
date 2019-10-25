import os
import nltk
import json
import argparse
import pickle
from collections import Counter

def loadcap(capdir):
    cap=os.path.join(capdir, 'captions.txt')
    cap_dict={}
    with open(cap) as cap:
        for line in cap:
            dic=json.loads(line)
            for i, j in dic.items():
                cap_dict[i]=j
    return cap_dict

class vocab_build():
    def __init__(self, cap_dict, threshold):
        self.word2id = {}
        self.id2word = {}
        self.index = 0
        self.build(cap_dict, threshold)

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.index
            self.id2word[self.index] = word
            self.index += 1

    def build(self, captions_dict, threshold):
        counter = Counter()
        tokens = []
        # get all words from the caption file
        for k, captions in captions_dict.items():
            for caption in captions:
                tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        counter.update(tokens)
        # set a number to discard words that appear less than the threshold
        words = [word for word, count in counter.items() if count >= threshold]

        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<pad>')
        for word in words:
            self.add_word(word)

    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.word2id['<unk>']
        
    def get_sentence(self, ids_list):
        sent = ''
        for cur_id in ids_list:
            cur_word = self.id2word[cur_id.item()]
            sent += ' ' + cur_word
            if cur_word == '<end>':
                break
        return sent

    def get_word(self, index):
        return self.id2word[index]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='train')
    parser.add_argument('-i', type=int, default=6) # threshold
    args=parser.parse_args()
    print('loading captions from folder: '+ str(args.dir) + '\nthreshold is: ' + str(args.i) )
    dir=args.dir
    threshold= args.i

    #load caption file
    cap_dict=loadcap(dir)

    ''''# get all words from the caption file
    counter = Counter()
    tokens = []
    for k, captions in cap_dict.items():
        for caption in captions:
            tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
    counter.update(tokens)
    # set a number to discard words that appear less than the threshold
    words = [word for word, count in counter.items() if count >= threshold]

    # add to the word list
    word2ind = {}
    ind2word = {}
    count = 0

    word2ind, ind2word, count = incr_word(word2ind, ind2word, '<unk>', count)
    word2ind, ind2word, count = incr_word(word2ind, ind2word, '<pad>', count)
    word2ind, ind2word, count = incr_word(word2ind, ind2word, '<start>', count)
    word2ind, ind2word, count = incr_word(word2ind, ind2word, '<end>', count)
    word2ind, ind2word, count = incr_word(word2ind, ind2word, words, count)'''

    Vocab=vocab_build(cap_dict, threshold)
    # save vocab data
    with open(os.path.join(args.dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(Vocab, f)
        print('vocab stored')

    print('total words stored:' + str(Vocab.index))
