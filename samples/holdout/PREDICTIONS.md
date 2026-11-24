# Blind-test predictions

Fretly's output for the hidden clips, written and committed **before** the
answers were seen, so the score can't be tuned after the fact. The clips were
recorded as `New Recording 17–19.m4a` (blind1 = 17, blind2 = 18, blind3 = 19).
Generated with the code at the parent commit by:

```
python scripts/make_tab.py ../samples/holdout/blindN.m4a --method both
```

The optimizer and the baseline gave identical tabs on all three.

## blind1

```
Notes: 12 detected

e|-------------------------|
B|---------------------0---|
G|-2-----------------------|
D|---3-0-0---------3-------|
A|-------------0-3---3---2-|
E|---------0-0-------------|
```

## blind2

```
Notes: 11 detected

e|-7-9-3-----0---------|
B|---------3-----------|
G|-------0-----3-------|
D|---------------2-4---|
A|-------------------4-|
E|---------3-----------|
```

## blind3

```
Notes: 11 detected, removed 1 ringing open string
  ringing open string: D3 at 1.95 s

e|-------------------|
B|-0---0-2---0-------|
G|---1---------2-1---|
D|-0-------0-------4-|
A|-------------------|
E|-------------------|
```

## After the answers (added 2026-11-23)

The predictions above are unchanged. Once the answers arrived, the clips were
renamed to describe what's played, and each has a reference tab next to it:

| Above | Recording | Now |
|---|---|---|
| blind1 | New Recording 17 | `low_walk` |
| blind2 | New Recording 18 | `cross_descent` |
| blind3 | New Recording 19 | `pos4_riff` |
