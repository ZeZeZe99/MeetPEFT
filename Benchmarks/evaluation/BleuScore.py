from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
class CocoScorer():
    def __init__(self, ref, gt):
        # param:
        # ref: list[str]
        # gt: list[str]
        assert len(gt) == len(ref)
        self.ref, self.gt = dict(), dict()
        for i in range(len(gt)):
            self.ref[i] = [ref[i]] #dict[key] = str
            self.gt[i] = [gt[i]]

        #print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
            #(Meteor(), "Meteor")
            #(Cider(), "CIDEr")
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            #print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                #print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores
