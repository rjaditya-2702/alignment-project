# CLadder rewards (keep yours, it's well-calibrated):
#     Step 1 (graph):        11
#     Step 2 (query type):   15
#     Step 3 (derivation):   24   ← LLM judge
#     Step 4 (computation):  30
#     Step 5 (answer):       20
#     Penalty: -100 per failed step, cascading

# CauSciBench rewards (different weights, softer penalties):
#     Step 1 breakdown:
#         treatment match:    5
#         outcome match:      5
#         control overlap:   15   ← this is the bottleneck
#         instrument/rv/tv:   5   ← method-specific variable
#     Step 2 (method):       30   ← biggest bottleneck
#     Step 3 (specification): 0   ← no ground truth, skip
#     Step 4 (code runs):    10
#     Step 5 (answer):       30   ← relative error, continuous
#     Penalty: -50 per failed step, NO cascading

