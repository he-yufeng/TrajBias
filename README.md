# TrajBias Coding

TrajBias studies how LLM-based evaluators score coding-agent trajectories under controlled changes to the information they receive.

## Research status

This is a research manuscript, not an accepted NeurIPS Datasets and Benchmarks paper or an accepted ACL/EMNLP paper. The earlier NeurIPS venue in this README's citation was an incorrect placeholder, not a publication record. The most recent recorded ARR cycle ended through a review-capacity lottery without scientific reviews.

This public repository is an early code snapshot. It is not the current manuscript, a fully open benchmark release, or a complete reproduction package. Do not infer current results or submission status from the old script names or historical model aliases.

## Scope of the evidence

The study uses matched trajectory comparisons to distinguish score sensitivity from verified evaluation quality. Outcome labels, inserted error-and-recovery exchanges, error placement and a limited redundant-message manipulation address different questions and should not be combined into a universal bias claim.

The previous README overstated some findings. In particular, lower scores after error-recovery insertion do not by themselves establish an inappropriate penalty, and a limited length manipulation does not show that verbosity never matters. Paired dominance is not interchangeable with independent-sample Cliff's delta. Historical judge aliases also do not authenticate request-level provider snapshots.

For those reasons, the old headline effect sizes and blanket findings have been removed. This correction does not report a new experiment or release unpublished scores.

## What is in this repository

The `scripts/` directory contains early research scripts for perturbation generation, judge evaluation, statistical analysis, plotting and data-export preparation. Their presence does not establish that an exported dataset is available, rights-cleared or sufficient to reproduce the current manuscript. This README no longer presents an unverified dataset link as a complete benchmark release.

Some scripts can make billable model requests. Inspect their configuration, service terms and output paths before running them. This repository is not an invitation to replay unknown or interrupted requests.

## Data and reuse

The current full trajectory and API-output archive is not cleared for upload or public redistribution. Public upstream repository text can retain incidental identifiers and third-party licensing obligations. Confidential reviewer companions are separate materials and carry no public redistribution permission.

The existing MIT licence for code already released here is unchanged. This notice does not revoke permissions already granted for that code, and it does not grant new permissions over trajectories, provider outputs or confidential reviewer materials.

## Citation

Until there is a verifiable publication record, cite the project as a research manuscript rather than conference proceedings:

```bibtex
@misc{he2026trajbias,
  title={TrajBias-Coding: Outcome-Conditioned Score Sensitivity in LLM Evaluation of Coding-Agent Trajectories},
  author={He, Yufeng},
  year={2026},
  note={Research manuscript; not a conference acceptance},
  howpublished={Public early-code repository},
  url={https://github.com/he-yufeng/TrajBias}
}
```
