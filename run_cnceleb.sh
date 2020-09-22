#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# cnceleb trials
cnceleb_root=data/CN-Celeb
eval_trails_core=data/eval_test/trials/trials.lst
nnet_dir=exp/xvector_nnet_cnceleb

stage=0
if [ $stage -le 0 ]; then
  # Prepare the CN-Celeb dataset. The script is used to prepare the development
  # dataset and evaluation dataset.
  local/make_cnceleb.sh $cnceleb_root data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train eval_enroll eval_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 2 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 8 --cmd "$train_cmd" \
    data/train data/train_no_sil exp/train_no_sil
  utils/fix_data_dir.sh data/train_no_sil

  # # Now, we need to remove features that are too short after removing silence
  # # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/train_no_sil/utt2num_frames data/train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_no_sil/utt2num_frames.bak > data/train_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_no_sil/utt2num_frames data/train_no_sil/utt2spk > data/train_no_sil/utt2spk.new
  mv data/train_no_sil/utt2spk.new data/train_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=5
  awk '{print $1, NF-1}' data/train_no_sil/spk2utt > data/train_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_no_sil/spk2num | utils/filter_scp.pl - data/train_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
  mv data/train_no_sil/spk2utt.new data/train_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_no_sil/spk2utt > data/train_no_sil/utt2spk

  utils/filter_scp.pl data/train_no_sil/utt2spk data/train_no_sil/utt2num_frames > data/train_no_sil/utt2num_frames.new
  mv data/train_no_sil/utt2num_frames.new data/train_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_no_sil
fi

local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/train_combined --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

if [ $stage -le 3 ]; then
  # Note that there are over one-third of the utterances less than 2 seconds in our training set,
  # and these short utterances are harmful for PLDA training. Therefore, to improve performance 
  # of PLDA modeling and inference, we will combine the short utterances longer than 5 seconds.
  utils/data/combine_short_segments_new.sh --speaker-only true \
    data/train 5 data/train_comb
  # Compute the energy-based VAD for train_comb
  sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
    data/train_comb exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train_comb
fi

if [ $stage -le 4 ]; then

  # Extract xvectors for cn-celeb data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G" --nj 8 \
    $nnet_dir data/train_comb \
    exp/xvectors_train_comb

  # The enroll and test data
  for name in eval_enroll eval_test; do
    sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 8 \
      $nnet_dir data/$name \
      exp/xvectors_$name
  done
fi

if [ $stage -le 5 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_train_comb/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_train_comb/xvector.scp \
    exp/xvectors_train_comb/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/xvectors_train_comb/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_train_comb/xvector.scp ark:- |" \
    ark:data/train_comb/utt2spk exp/xvectors_train_comb/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd exp/xvectors_train_comb/log/plda.log \
    ivector-compute-plda ark:data/train_comb/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_train_comb/xvector.scp ark:- | transform-vec exp/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_train_comb/plda || exit 1;
fi

if [ $stage -le 6 ]; then
  # Compute PLDA scores for CN-Celeb eval core trials
  $train_cmd exp/scores/log/cnceleb_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_eval_enroll/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_train_comb/plda - |" \
    "ark:ivector-mean ark:data/eval_enroll/spk2utt scp:exp/xvectors_eval_enroll/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_train_comb/mean.vec ark:- ark:- | transform-vec exp/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_train_comb/mean.vec scp:exp/xvectors_eval_test/xvector.scp ark:- | transform-vec exp/xvectors_train_comb/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$eval_trails_core' | cut -d\  --fields=1,2 |" exp/scores/cnceleb_eval_scores || exit 1;

  # CN-Celeb Eval Core:
  # EER: 
  # minDCF(p-target=0.01): 
  # minDCF(p-target=0.001): 
  echo -e "\nCN-Celeb Eval Core:";
  eer=$(paste $eval_trails_core exp/scores/cnceleb_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores/cnceleb_eval_scores $eval_trails_core 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi
