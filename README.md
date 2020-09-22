# xvector-cnceleb
kaldi based x-vector trained on Cn-Celeb database.
# Kaldi configuration
export KALDI_ROOT=/kaldi/dir
# Usage
bash run_cnceleb.sh
# Result
- 1、x-vector(cn-celeb) + PLDA(cn-celeb)
  - CN-Celeb Eval Core:EER: 16.71% **** minDCF(p-target=0.01): 0.7657 **** minDCF(p-target=0.001): 0.8823

- 2、x-vector(voxceleb) + PLDA(cn-celeb)
  - CN-Celeb Eval Core:EER: 12.43% **** minDCF(p-target=0.01): 0.6064 **** minDCF(p-target=0.001): 0.7381
 # References:
 - https://www.danielpovey.com/files/2018_icassp_xvectors.pdf(X-VECTORS)
 - https://arxiv.org/pdf/1911.01799.pdf(Cn-Celeb)
 - Voxceleb database:http://openslr.org/49/
 - 开源voxceleb模型:https://kaldi-asr.org/models/m7
 - CN-Celeb database:http://openslr.org/82/

