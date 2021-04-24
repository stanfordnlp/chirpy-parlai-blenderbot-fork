#!/usr/bin/env python3
import logging
from typing import List, Tuple

import torch

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import (
    GlobalAverageMetric,
    GlobalTimerMetric,
)
from parlai.core.torch_agent import Output
from parlai.core.message import Message
from parlai.utils.distributed import is_primary_worker
from parlai.utils.misc import warn_once


class CgenAgent(TransformerGeneratorAgent): # CGen = Conditional GENeration -- i.e. based on prefix. Sorry, PARLAI parsing means I have to make a jank name.

    def act(self, prefix_text=None):
        """
        Call batch_act with the singleton batch.
        """
        # BatchWorld handles calling self_observe, but we're in a Hogwild or Interactive
        # world, so we need to handle this ourselves.
        if prefix_text:
            prefix_tensor = torch.LongTensor(self.history.parse(prefix_text)).unsqueeze(0)
        else:
            prefix_tensor = None
        response = self.batch_act([self.observation], prefix_tensor)[0]
        return response

    def batch_act(self, observations, prefix_tensor=None):
        """
        Process a batch of observations (batchsize list of message dicts).
        These observations have been preprocessed by the observe method.
        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)
        self.global_metrics.add('exps', GlobalTimerMetric(batch.batchsize))

        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            lt = (batch.label_vec != self.NULL_IDX).sum().item()
            ltpb = GlobalAverageMetric(lt, float(is_primary_worker()))
            self.global_metrics.add('ltpb', ltpb)
            self.global_metrics.add('ltps', GlobalTimerMetric(lt))

            ct = (batch.text_vec != self.NULL_IDX).sum().item()
            ctpb = GlobalAverageMetric(ct, float(is_primary_worker()))
            self.global_metrics.add('ctpb', ctpb)
            self.global_metrics.add('ctps', GlobalTimerMetric(ct))

            ttpb = GlobalAverageMetric(ct + lt, float(is_primary_worker()))
            self.global_metrics.add('tpb', ttpb)
            self.global_metrics.add('tps', GlobalTimerMetric(ct + lt))

        if self.is_training:
            # register the start of updates for later counting when they occur
            self.global_metrics.add('ups', GlobalTimerMetric(0))
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch, prefix_tensor=prefix_tensor)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # register the end of timers
        endtimer = GlobalTimerMetric(0)
        self.global_metrics.add('exps', endtimer)
        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            self.global_metrics.add('ltps', GlobalTimerMetric(0))
            self.global_metrics.add('ctps', GlobalTimerMetric(0))
            self.global_metrics.add('tps', GlobalTimerMetric(0))

        return batch_reply


    def eval_step(self, batch, prefix_tensor=None):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None and batch.image is None:
            return
        if batch.text_vec is not None:
            bsz = batch.text_vec.size(0)
        else:
            bsz = len(batch.image)
        self.model.eval()
        cand_scores = None
        token_losses = None

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss, model_output = self.compute_loss(batch, return_output=True)
            if self.output_token_losses:
                token_losses = self._construct_token_losses(
                    batch.label_vec, model_output
                )

        preds = None
        if self.skip_generation:
            warn_once("--skip-generation true produces limited metrics")
        else:
            maxlen = self.label_truncate or 256
            dev = 'cpu' if self.opt["no_cuda"] else 'cuda' 
            beam_preds_scores, beams = self._generate(batch.to(dev),
                    self.beam_size,
                    maxlen,
                    prefix_tokens=prefix_tensor.to(dev) if prefix_tensor is not None else None)
            preds, scores = zip(*beam_preds_scores)
            self._add_generation_metrics(batch, preds)

            # bsz x beamsize
            beam_texts: List[List[Tuple[str, float]]] = []
            for beam in beams:
                beam_texts.append([])
                for tokens, score in beam.get_rescored_finished():
                    try:
                        beam_texts[-1].append((self._v2t(tokens), score.item()))
                    except KeyError:
                        logging.error("Decoding error: %s", tokens)
                        continue

        cand_choices = None
        cand_scores = None
        if self.rank_candidates:
            cand_choices, cand_scores = self.rank_eval_label_candidates(batch, bsz)

        text = [self._v2t(p) for p in preds] if preds is not None else None
        if text and self.compute_tokenized_bleu:
            # compute additional bleu scores
            self._compute_fairseq_bleu(batch, preds)
            self._compute_nltk_bleu(batch, text)
        retval = Output(
            text, cand_choices, token_losses=token_losses, cand_scores=cand_scores
        )
        if not self.skip_generation:
            retval.beam_texts = beam_texts
        return retval
