# Original-data diagnostic and revised dataset recommendation

Checked 11 October 2026 after clarification that the professor requested new data because the models achieved very high training and validation accuracy. This is an exploratory diagnostic on the existing validation partition, not an independent test evaluation.

## Saved model results

| Model | Final training accuracy | Best validation accuracy | Correct / validation total |
|---|---:|---:|---:|
| CNN from scratch | 99.43% | 97.78% | 44/45 |
| ViT-Tiny from scratch | 94.89% | 95.56% | 43/45 |
| Pretrained DINOv2 classifier | 95.45% | 95.56% | 43/45 |
| Pretrained VMamba classifier | 97.73% | 97.78% | 44/45 |

Sources: each model's saved `accuracy_by_epoch.csv` and `best_accuracy_metrics.json`. Final training accuracy and best validation accuracy can come from different epochs.

The shared manifest has 176 training and 45 validation records, with no independent test set. All 221 original grayscale BMPs have shape 760×770, distinct filenames and distinct pixel arrays. No exact image duplicates span the partitions. This does not rule out near-duplicates, common generation-history groups or other shortcuts.

## Average-brightness diagnostic

For each original image, I calculated its mean grayscale intensity. On the 176 training records only, I searched thresholds halfway between consecutive distinct intensity values and both classification directions, maximizing training accuracy. The selected rule is:

```text
Predict High if mean grayscale intensity <= 202.30406698564593
Otherwise predict Low
```

It classifies 170/176 training images correctly (96.59%) and 43/45 validation images correctly (95.56%). There was no threshold tuning on validation. This simple rule matches the saved best validation accuracy of ViT-Tiny and DINOv2. A strong global-intensity cue is therefore sufficient for high accuracy on this partition; the neural-network scores alone do not demonstrate reliance on complex spatial relationships.

The class-average intensities are approximately 218.004 for Low and 190.345 for High. Brightness can be physically correlated with material coverage or can reflect image rendering. Its predictive value is not proof of data leakage; its physical cause was not established here.

The [original paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adpr.202500284) kept 221 examples from 329 generated devices and discarded intermediate-bandwidth examples. It defined Low and High with separated normalized-bandwidth thresholds. This could make classification easier, but the diagnostic does not establish causation.

One validation mistake changes accuracy by 2.22 percentage points. Models are selected using the same validation partition on which their best scores are reported. An untouched test set would strengthen future generalization claims. The user subsequently chose training and validation only for the SUTD pilot, so that run does not evaluate test data.

## Revised recommendation

Use [SUTD-PRCM RDN128](https://github.com/veya2ztn/SUTD_PRCM_dataset) as the first new benchmark. Its [paper](https://arxiv.org/html/2203.00002) describes a difficult binary maximum-reflection-peak task, with the tested conventional networks below 60% and its architecture-search model above 75%. Those are results under the authors' protocol, not predictions for these four models. The local compact-release audit found 111,000 unique layouts and approximately balanced labels.

The downloaded compact B1N branch is processed by the author's loader as `1 - abs(first complex branch)`, rounded to three decimals and sampled at 128 bins. Our actual experiment target is the fixed `argmax(released compact response) >= 83` rule. We do not invert this response again or equate its argmax with the raw-reflection peak or a particular GHz threshold.

For comparison, I fitted the same type of average-brightness threshold on SUTD RDN128's 108,000 training records, using the fixed `argmax(128-bin spectrum) >= 83` labels. Its best training accuracy was only 51.50%, close to the 51.26% majority-class training baseline. I did not evaluate this diagnostic on the reserved SUTD test partition. This suggests that the particular global-intensity cue found in the original dataset is much less useful for this SUTD task; it does not establish the performance of richer features or neural models.

Keep the model backbones, two-class heads and classification loss. Change dataset loading, image preparation, class names, sample-count assumptions and split/evaluation handling. Reserve the released test partition, derive validation from the released training partition, and use one fixed label rule across all partitions. Native 16×16 images should be preserved before resizing; the paper's physical domain is microwave metasurfaces.

Use UCLA Explainability for Photonics as a later optical experiment if needed. A newly defined high/low absorption label may itself have an easy shape or coverage cue, so it is not yet established as a stronger classification challenge.

Predefine tasks, splits and model-selection rules. Assess a simple brightness/fill-fraction baseline and the four models, and report results even if accuracy remains high. Choosing tasks or thresholds after looking for lower validation accuracy would bias the comparison.
