import numpy as np
from collections import defaultdict, Counter
import logging
logger = logging.getLogger()

def softmax(w):
    e = np.exp(w)
    dist = e / np.sum(e, axis=0)
    return dist


def joint(char, y, input_dist, alphabet, B, Pr, pb_):
    if y != '' and char == y[-1]:
        out = input_dist[alphabet.index(char)] + pb_[B.index(y)]
    else:
        out = input_dist[alphabet.index(char)] + Pr[B.index(y)]
    return out

def ctc_beamsearch(input_dist, alphabet='-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=10):
    # Initialization
    input_dist = np.exp(input_dist)
    num_time_steps = input_dist.shape[0]
    best_beams = ['']
    Pnb = defaultdict(Counter)
    Pb = defaultdict(Counter)
    Pb[0][''] = 1
    Pnb[0][''] = 0
    input_dist = np.vstack((np.zeros(input_dist.shape[1]), input_dist))
    # Iterate over time steps
    for t in range(1, num_time_steps):
        # Iterate over most probable beams from previous step
        for beam in best_beams:
            # Iterate over all characters
            high_prob_alphabet = [alphabet[i] for i in np.where(input_dist[t] > 0.1)[0]]
            # high_prob_alphabet = alphabet
            for character in high_prob_alphabet:
                # Extend beams by adding blankx
                if character == '-':
                    Pb[t][beam] += input_dist[t, 0] * (Pb[t - 1][beam] + Pnb[t - 1][beam])
                # Extend beams by adding non-blank character
                else:
                    beam_plus = beam + character
                    # Extend beams by adding the last character of non-blank beams
                    if len(beam) > 0 and character == beam[-1]:
                        Pnb[t][beam] += Pnb[t - 1][beam] * input_dist[t, alphabet.index(character)]
                        Pnb[t][beam_plus] += Pb[t - 1][beam] * input_dist[t, alphabet.index(character)]
                    else:
                        Pnb[t][beam_plus] += (Pb[t - 1][beam] + Pnb[t - 1][beam]) * input_dist[t, alphabet.index(character)]

        all_beams = Pnb[t] + Pb[t]
        best_beams = sorted(all_beams, key=lambda x: all_beams[x], reverse=True)[:k]
    all_beams = [(k, v) for k, v in sorted(all_beams.items(), key=lambda x: x[1], reverse=True)]
    return all_beams[:k]


def ctc_beamsearch_log(input_dist, alphabet='-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=10):
    # Initialization
    num_time_steps = input_dist.shape[0]
    best_beams = ['']
    Pnb = defaultdict(lambda: -1e5)
    Pnb_current = defaultdict(lambda: -1e5)
    Pb = defaultdict(lambda: -1e5)
    Pb_current = defaultdict(lambda: -1e5)
    Ptot = defaultdict(lambda: -1e5)
    Pb[''] = -0.0001
    Pnb[''] = -1e5
    # Ptot[''] = np.logaddexp(Pb[''], Pnb[''])
    zero_step = np.full_like(input_dist[0], fill_value=-1e5)
    input_dist = np.vstack((zero_step, input_dist))
    # Iterate over time steps
    for t in range(1, num_time_steps):
        # Iterate over most probable beams from previous step
        for beam in best_beams:
            # Iterate over all characters
            high_prob_alphabet = [alphabet[i] for i in np.where(input_dist[t] > -2.3)[0]]
            # high_prob_alphabet = alphabet
            for character in high_prob_alphabet:
                # Extend beams by adding blank
                if character == '-':
                    Pb_current[beam] = np.logaddexp(Pb_current[beam], input_dist[t, 0] + np.logaddexp(Pb[beam], Pnb[beam]))
                    Ptot[beam] = np.logaddexp(Pb_current[beam], Pnb_current[beam])
                else:
                    beam_plus = beam + character
                    # Extend beams by adding the last character of non-blank beams
                    if len(beam) > 0 and character == beam[-1]:
                        Pnb_current[beam] = np.logaddexp(Pnb_current[beam], Pnb[beam] + input_dist[t, alphabet.index(character)])
                        Ptot[beam] = np.logaddexp(Pb_current[beam], Pnb_current[beam])
                    else:
                        Pnb_current[beam_plus] = np.logaddexp(Pnb_current[beam_plus],
                                                              Pnb[beam] + input_dist[t, alphabet.index(character)])
                    # Extend beams by adding non-blank character
                    Pnb_current[beam_plus] = np.logaddexp(Pnb_current[beam_plus],
                                                          Pb[beam] + input_dist[t, alphabet.index(character)])
                    Ptot[beam_plus] = np.logaddexp(Pb_current[beam_plus], Pnb_current[beam_plus])

        Pb = Pb_current
        Pnb = Pnb_current
        Pb_current = defaultdict(lambda: -1e5)
        Pnb_current = defaultdict(lambda: -1e5)
        all_beams = Ptot
        Ptot = defaultdict(lambda: -1e5)
        best_beams = sorted(all_beams, key=lambda x: all_beams[x], reverse=True)[:k]

    all_beams = [(k, v) for k, v in sorted(all_beams.items(), key=lambda x: x[1], reverse=True)]
    return all_beams[:k]


def ctc_beamsearch_old(input_dist, alphabet='-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=10):
    # input_dist should be numpy matrix
    # beamsize is k
    T = input_dist.shape[0]
    B = ['']
    Pr = [0]
    pnb_ = [-1e10]
    pb_ = [0]
    for t in range(T):
        # get k most probable sequences in B
        # print 't is ' + str(t)
        B_new = []
        Pr_new = []
        pnb_new = []
        pb_new = []
        ind = np.argsort(Pr)[::-1]
        B_ = [B[i] for i in ind[:k]]
        for y in B_:
            # print 'y is ' + y
            if y != '':
                pnb = pnb_[B.index(y)] + input_dist[t, alphabet.index(y[-1])]
                if y[:-1] in B_:
                    pnb = np.logaddexp(pnb, joint(y[-1], y[:-1], input_dist[t], alphabet, B, Pr, pb_))
            else:
                pnb = -1e10

            pb = Pr[B.index(y)] + input_dist[t, alphabet.index('-')]

            B_new += [y]
            Pr_new += [np.logaddexp(pnb, pb)]
            pnb_new += [pnb]
            pb_new += [pb]
            for c in alphabet[1:]:
                # print 'c is ' + c
                pb = -1e10
                pnb = joint(c, y, input_dist[t], alphabet, B, Pr, pb_)
                pb_new += [pb]
                pnb_new += [pnb]
                Pr_new += [np.logaddexp(pnb, pb)]
                B_new += [y + c]
        B = B_new;
        Pr = Pr_new;
        pnb_ = pnb_new;
        pb_ = pb_new;

    # out_ind = np.argmax([Pr[i]/len(B[i]) if len(B[i]) > 0 else -1e10 for i in range(len(B))])
    #    out_ind = np.argmax([Pr[i] for i in range(len(B))])
    #    B_unique = list(set(B))
    res = {}
    for i in range(len(B)):
        if B[i] not in res:
            res[B[i]] = Pr[i]

    res = [(k, v) for k, v in sorted(res.items(), key=lambda x: x[1])]
    res = res[-1:-k:-1]

    prob = [_[1] for _ in res]
    prob = softmax(prob)
    res = [(res[i][0], prob[i]) for i in range(len(res))]
    return res


if __name__ == '__main__':
    # for i in range(5):
    #     dist = np.load('test{}.npy'.format(i))
        # print(diff[np.abs(diff) > 1])
        # diff[diff > 1] = 0
    dist = np.load('test2.npy')
    logger.info('CTC beamsearch probability = {}'.format(ctc_beamsearch(dist)))
    logger.info('CTC beamsearch logarithm = {}'.format(ctc_beamsearch_log(dist)))
    # dist = np.array([[0.2, 0.3, 0.5],
    #           [0.4, 0.4, 0.2],
    #           [0.1, 0.3, 0.6],
    #           [0.5, 0.2, 0.3]])
    # print(ctc_beamsearch(log_dist, '-ab', k=5))
    print('Done')
