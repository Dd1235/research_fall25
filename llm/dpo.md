# Pre-training LLMs

- usually self surpervision
- let is auto complete
- RLHF is the most common way, base llm trained on large corpora, let llm produce pairs of outputs. a copy of hte llm becomes hte reward model that by learning to mimic human ratings, llm trains to receive high feedback from the reward model.
- rlhf is unstable and needs to train the reward model so is large.

- wo reward model and without rl? just with cross entropy loss? -> dpo

- in dpo llm trains to assign high probability to positive examples and low probability to negative examples.

human rating is not llm output, how to compare output of model to expected output? can't use usual supervised learning. Now how will loss function work? Without loss value how to run backprop? Cannot compute gradient of model wrt human feedback. So we used RL tricks. Short coming: also need positive negative pairs. with rlhf, reward model can create more data by labelling outputs as positive or negative.
