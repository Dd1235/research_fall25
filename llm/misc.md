- You can get smarter models either by increasing the parameters, or letting it "think", ie, ues the same data point more judiciously. The latter is more efficient, and saves more money. If you generate a really long response with more chain of thought that is better.

- You don't have to update every weight, find the "parts" that were affected with this data point, and only update those. This saves cost as you generate more response wtih one datapoint.

- Find the entropy/surprise (titans:learning to memorize at test time), unless you have information in the datapoint, you don't have to update the weights. Data != information.

- LLM is made of transformers, attention is O(n^2) operation, LoLCATs, use different operators to move away from attention to a new operation that approximates attention but is O(n)

- Neuromorphic computing
