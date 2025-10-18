Here are bite-size, copy-pasteable examples of `re.findall` (each shows the output you’ll get):

```python
import re
```

1. All numbers

```python
s = "Order 12 costs 9.50, order 7 costs 3"
re.findall(r"\d+(?:\.\d+)?", s)  # ['12', '9.50', '7', '3']
```

2. Words starting with a capital letter

```python
s = "Alice and bob met Carol in paris"
re.findall(r"\b[A-Z][a-z]+\b", s)  # ['Alice', 'Carol']
```

3. Email domains (note: returns just the group)

```python
s = "a@x.com, b@y.co.uk"
re.findall(r"@([\w.-]+\.\w+)", s)  # ['x.com', 'y.co.uk']
```

4. Capture date parts (groups → list of tuples)

```python
s = "Due: 2025-10-18 and 1999-01-01"
re.findall(r"(\d{4})-(\d{2})-(\d{2})", s)  # [('2025','10','18'), ('1999','01','01')]
```

5. Text between double quotes (non-greedy)

```python
s = 'say "hi" then "bye"'
re.findall(r'"(.*?)"', s)  # ['hi', 'bye']
```

6. Start of each line (multiline flag)

```python
s = "first line\nsecond line"
re.findall(r"^\w+", s, flags=re.M)  # ['first', 'second']
```

7. Case-insensitive search

```python
s = "Cat scatters cATs"
re.findall(r"cat", s, flags=re.I)  # ['Cat', 'cat']
```

8. Overlapping matches via lookahead (since `findall` itself doesn’t overlap)

```python
s = "ababa"
re.findall(r"(?=(aba))", s)  # ['aba']  # (positions 0 and 2; only one match fits here)
```

9. Repeated word (backreference)

```python
s = "this is is a test test"
re.findall(r"\b(\w+)\s+\1\b", s)  # ['is', 'test']
```

Tip:

- If your pattern has groups, `findall` returns the captured groups; no groups → it returns the full matches.
- For DOT to match newlines, add `flags=re.S`.
