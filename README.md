# PGsheetGen

This repository accompanies the work introducing PGsheet, and proposes
PGsheetGen, a system for automatically generating PGsheets from property
graph data using large language models (LLMs).

## Overview

Scripts for generating, sampling, and analyzing property graph datasets
(DBLP, LDBC, and Knows) using LLM-based prompting strategies.

## Repository Structure

Scripts/
  prompting_gpt.py                  - GPT-based prompting for LDBC and Knows dataset
  prompting_gpt_dblp.py             - GPT prompting on DBLP data
  prompting_llm_dblp_ISR.py         - ISR strategy on DBLP
  prompting_llm_dblp_MPR.py         - MPR strategy on DBLP
  prompting_llm_dblp_PR.py          - PR strategy on DBLP
  prompting_llm_MPR.py              - MPR strategy for LDBC and Knows data
  prompting_ISR_LDBC_Knows.py       - ISR strategy on LDBC/Knows
  prompting_PR_LDBC_Knows.py        - PR strategy on LDBC/Knows
  serialization_DBLP.py             - Serialize DBLP graph data
  serialization_LDBC.py             - Serialize LDBC graph data
  serialization_Knows.py            - Serialize Knows graph data
  serialization_fast_DBLP.py        - Faster DBLP serialization
  stratified_sampling.py            - Stratified sampling of dataset
  shuffling.py                      - Shuffle dataset entries

data_analysis/
  analyze.py
  analyze2.py
  analyze3.py

Data/
Contains initial PGsheet template and final PGsheet templates by each models across 3 runs.

## License

See LICENSE.
