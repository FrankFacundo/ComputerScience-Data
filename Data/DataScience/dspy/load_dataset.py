from dspy.datasets import HotPotQA

dataset = HotPotQA(
    train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0
)

trainset = [x.with_inputs("question") for x in dataset.train]

len(trainset)
