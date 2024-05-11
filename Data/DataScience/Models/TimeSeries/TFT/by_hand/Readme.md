train row example: `train.iloc[0]`

```text
power_usage                          -0.127174
t                                      26304.0
days_from_start                           1096
categorical_id                               0
date                       2014-01-01 00:00:00
id                                      MT_001
hour                                 -1.661325
day                                          1
day_of_week                          -0.499719
month                                        1
hours_from_start                     -1.731721
categorical_day_of_week                      2
categorical_hour                             0
Name: 17544, dtype: object
```

train_dataset row example: `train_dataset[0][0][0]`

```text
power_usage                          -0.127174
hour                                 -1.661325
day_of_week                          -0.499719
hours_from_start                     -1.731721
categorical_id                             0
```

---

```python
train_dataset[0] : tuple = (data_map["inputs"], data_map["outputs"], data_map["active_entries"])
```
