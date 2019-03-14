__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
from .build_vocab import Vocabulary # Need to manually import the class


class COCOSpiceScorer:
    def __init__(self, coco, vocab):
        self.coco = coco
        self.vocab = vocab
        self.vocab_filter = [self.vocab('<start>'), self.vocab('<end>')]    # Words to filter out
        self.unk_tok = '<unk>'
        self.tokenizer = PTBTokenizer()
        self.spice = Spice()
        self.image_ids = coco.getImgIds()

        # Gather image captions and caption
        self.gts = {id_: self.coco.imgToAnns[id_] for id_ in self.image_ids}

        # Tokenize the reference caps here for speed
        self.gts_tokens = self.tokenizer.tokenize(self.gts)

    def image_id_to_annotations(self, img_ids):
        """
        Return the tokenized reference
        annotations for each of img_id
        :param img_ids: [img_id1, img_id2, img_id3,...]
        :return: {id1: [REF1, REF2, REF3, ...], id2: [...]}
        """
        return {id_: self.gts_tokens[id_] for id_ in img_ids}

    def tokenify_index_seq(self, index_seq):
        """
        Convert a list of vocabulary indices to a string sentence
        :param index_seq: [1, 2, 3, 7, 9, 10] list of integer vocab indices
        :return: INF_CAP_STRING
        """
        tok_seq = [self.vocab.idx2word.get(t, self.unk_tok)
                   for t in index_seq if t not in self.vocab_filter]
        seq = " ".join(tok_seq).strip()
        return seq

    def prepare_inf_batch(self, inf_batch):
        """
        Tokenify each index list and prepare for PTBTokenizer
        :param inf_batch: {id1: [4, 1, 2, 3, 5], id2: [4, 6, 5, 9, 5]}
        :return: {id1: [{'caption': 'abcdef...'}], ...}
        """
        batch_ = {k: [{'caption': self.tokenify_index_seq(v)}] for k, v in inf_batch.items()}
        batch_ = self.tokenizer.tokenize(batch_)
        return batch_

    def get_spice_score(self, inf_batch):
        """
        Compute the SPICE score for the batch.
        Input a list of vocabulary indices for each image ID
        :param inf_batch: {id1: [4, 1, 2, 3, 5], id2: [4, 6, 5, 9, 5]}
        :return: (average_score, scores):tuple
        """
        gts = self.image_id_to_annotations(inf_batch.keys())
        res = self.prepare_inf_batch(inf_batch)
        return self.spice.compute_score(gts, res)


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up tokenisation
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
            # (Meteor(), "METEOR") # Removed due to functionality bugs
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
