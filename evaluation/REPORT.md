# Meeting Scribe evaluation

The measured subset contains ten recordings totaling 19.37 minutes. Nine are read-speech clips. One is a 17 min 29 s English meeting with overlapping speakers. These are small comparisons on this laptop. Per-language differences also reflect the speakers, recording lengths and domains sampled.

## Speech recognition experiment

Both beam settings used the same decoded waveform, voice activity detector, multilingual Whisper small snapshot and float16 CTranslate2 runtime on an NVIDIA GeForce RTX 5060 Laptop GPU. Each configuration received a warm-up, then ran once per recording. Run order alternated across recordings. WER is total word edits divided by total reference words for each language. Scoring uses Unicode NFKC, case folding, punctuation/symbol replacement and whitespace tokenization. Hindi combining marks are retained. There is no stemming or number expansion.

| Source language | Recordings | Beam 1 WER | Beam 5 WER | Median RTF, beam 1 / 5 | Peak RSS MiB, beam 1 / 5 | Peak device usage MiB, beam 1 / 5 |
|---|---:|---:|---:|---:|---:|---:|
| English | 4 | 28.42% | 27.41% | 0.025 / 0.028 | 2441 / 2348 | 2581 / 2677 |
| Hindi | 3 | 48.39% | 44.09% | 0.068 / 0.084 | 1607 / 1608 | 2579 / 2579 |
| Spanish | 3 | 2.94% | 2.94% | 0.024 / 0.030 | 1613 / 1613 | 2579 / 2579 |

Real-time factor (RTF) divides ASR wall seconds by audio seconds. It covers VAD, language detection and consumption of all Whisper segments. File decoding, startup, summarization and HTTP are timed separately or excluded. Memory is sampled every 50 ms while both models are resident. RSS is the process working set. GPU usage is device-wide and includes unrelated allocations. A short-lived peak can fall between samples.

The wider beam reduced errors on AMI ES2004a from 812 to 782. It increased errors on English FLEURS clip 10052240106321793346 from 2 to 3. It tied on all three Spanish clips. This supports keeping beam 5 as a quality-oriented default on this machine, with limited evidence about other meetings. The English WER is dominated by the AMI transcript, which includes disfluencies and overlapping speech.

For speech translation, beam 5 was compared to aligned English references independently of ASR WER. Mean sentence chrF was **42.32/100 for Hindi** and **43.17/100 for Spanish**. The implementation uses character orders 1–6, beta 2, effective orders, whitespace removal, and case-sensitive text. chrF measures wording overlap and can penalize a valid paraphrase.

## Summary quality

Each system was scored against the same AMI annotator's abstract. ROUGE-1/2 use clipped unigram/bigram overlap, and ROUGE-L uses longest common subsequence. Values are F1 on a 0–1 scale with the WER token normalization and no stemming. The extractive baseline takes the first three sentences of the beam-5 ASR transcript.

| Input and method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Human transcript → BART | 0.152 | 0.013 | 0.089 |
| ASR transcript → BART | 0.141 | 0.024 | 0.082 |
| ASR transcript → first three sentences | 0.065 | 0.000 | 0.052 |

**Claim review.** The ASR-based summary correctly preserved the first-year €50 million profit goal and broad target audience. It turned the phrase about an international market into a named entity associated with the €25 price. Of three reviewed assertions (price attribution, profit target, audience), one contained an unsupported entity. It omitted the remote-control design goal and the discussion of menus and an LCD. With the human transcript as input, BART instead called the product a pen. These are reviewer judgments against the transcript. One meeting cannot establish a hallucination rate for the product.

**Reversal check.** A separate scripted meeting placed a tentative Friday proposal at the beginning and a decision to cancel Friday and ship Monday at 30:30, with more than one encoder window in between. The final output retained the change (**1/1 final decisions**) and added **0 unsupported assertions** in this fixture. The test requires a synthesis pass. The response can also preserve a missed explicit decision by quoting its source. This case does not establish that BART can reliably resolve implicit reversals. AMI ES2004a's annotator supplied no final decisions to score.

## Action and decision extraction

Sixteen hand-specified English sentences test direct assignments, commitments, predictions, negation, decisions and missing fields. They are authored regression examples. They have no independent annotator or representative class distribution. A true positive requires the right sentence and type. For the keyword baseline, a match on “decided” is treated as a decision and other matches as actions. The additional exact-match count also requires the labeled owner and due phrase, including nulls.

| Extractor | TP / FP / FN | Precision | Recall | F1 | Exact including fields out of 10 reference items |
|---|---|---:|---:|---:|---:|
| Original keyword baseline | 2 / 3 / 8 | 0.40 | 0.20 | 0.267 | 1 |
| Structured rules | 8 / 0 / 2 | 1.00 | 0.80 | 0.889 | 8 |

The structured rules missed the implicit request for someone to take minutes and the decision expressed as “signed off.” Read the [labeled cases](actions.json) alongside the [per-recording hypotheses, timings and scores](results.json).

## Application checks

- The supplied 29 min 58 s MP3 completed in **64.13 seconds** on the RTX 5060 after startup. The finished pipeline returned 426 timestamped segments and a 34-word synthesized summary. This is a processing check. That recording has no reference minutes.
- A forced CPU run on the 10.56-second English FLEURS clip returned two segments and a generated summary in **5.59 seconds** after startup.
- The live API passed original-language and translation requests for English, Hindi and Spanish, explicit silence, attenuated speech, malformed input, cancellation and saturation checks. The Python test server denied outbound non-loopback socket connections. Headless Chrome verified upload, refresh recovery, three exported file types and cancellation.
- A visible production build in headless Chrome with software WebGL made **804 draw calls in two seconds** during a simulated processing job. Repeating the measurement after switching that phase to on-demand rendering gave **zero steady-state draw calls**. This measures rendering work. A GPU inference speedup was not measured.

## Reproduction and attribution

Run `python evaluation/prepare.py` once with internet access from the project root, then `python evaluate.py` offline. The default manifest uses the first three distinct available aligned IDs in each selected FLEURS test audio archive, before inspecting model output. The AMI recording was selected for its human annotations. No local model training uses these recordings. Pretraining exposure is unknown.

- Google FLEURS, by Conneau and colleagues, supplies the English, Hindi and Latin American Spanish speech and human reference text. The preparation script selects test examples, pairs English references by sentence ID, and retains the source filenames. Dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. [Dataset and attribution](https://huggingface.co/datasets/google/fleurs) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- The AMI Project supplies ES2004a mixed-headset audio and manual annotations version 1.6.2. The preparation script joins speaker turns by start time and extracts the original annotator's abstract. This linearizes overlap for WER. [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) · [Corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

A custom JSON manifest contains a list of objects with `id`, relative `audio`, ISO language code `language`, and source-language `transcript`. Add an `english_reference` for translation scoring and an independently written English `summary` for ROUGE. The default report records model revisions, device, individual predictions and reference-word denominators. Use a new output path to preserve a comparison run.
