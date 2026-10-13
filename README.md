# Nova-Project

## What is this?

NOVA is an AI model that learns to **spot a hidden rule from just a few examples**, and then apply that rule to
something new.

For example, if you see:

- `2 4 6` → `4 8 12`
- `3 5` → `6 10`

you can guess the rule is "multiply by two", and answer `7 1 9` → `14 2 18`.
NOVA tries to do this kind of puzzle, and the real question is whether it still works when things get harder:
rules combined in new ways, much bigger inputs than it practised on, or a kind of rule it has never seen at all.

## The main idea

Most models look at all the examples and blend together whatever each one suggests.
NOVA does something closer to how a careful person works:

1. **Guess.** Each example gives its own guess at the rule.
2. **Cross-check.** Every guess is tried out on the *other* examples. A guess that only fits its own example is
   probably wrong. A guess that also explains the others is probably right.
3. **Trust the guesses that pass.** NOVA starts from a blend of the guesses that held up, and gives little weight to the rest.
   This also means one bad or misleading example does less damage.
4. **Improve.** It tests its current rule against the examples, sees where it goes wrong, and adjusts, a few times.
5. **Stop when it's good enough.** Easy puzzles stop early; harder ones get more attempts. So NOVA spends effort
   where it's needed.

The order of the examples doesn't matter: shuffle them and NOVA gives the same answer.

The plan for this project, including exactly what would count as NOVA *failing*, was written down before the
experiments are run. It's in [preregistration.md](preregistration.md).

## Where things stand

- ✅ The model, the comparison models, the puzzle generator and all the tools are built.
- ✅ 79 automatic checks pass, and a "planted bug" test shows those checks catch 8 out of 8 common mistakes.
- ⏳ The big experiments (which need a GPU for many hours) **have not been run yet**, so **no results are claimed here**.

## What you need

- Python (version 3.10 or newer)
- PyTorch, a free library for building AI models

Nothing else. The project deliberately avoids other add-on libraries, and an automatic check enforces this.

## Try it yourself

Download the project and install PyTorch:

```bash
git clone git@github.com:Vaibhav2520/Nova-Project.git
cd Nova-Project
pip install torch
```

**Run the automatic checks.** This takes about 20 seconds on an ordinary computer. A few checks that need a graphics
card (GPU) are skipped if you don't have one.

```bash
python -m unittest discover -s nova/tests -t .
```

**Run the planted-bug test.** This deliberately breaks the code in 8 different ways, one at a time, on a copy of the
project, and confirms the checks notice every one.

```bash
python tools/seed_bugs.py
```

**Run the experiments.** Use Google Colab, a free online notebook with access to GPUs:

1. Put the project folder anywhere in your Google Drive.
2. Open `NOVA_Research.ipynb` in Colab and choose a GPU runtime.
3. Run the cells from top to bottom. The notebook finds the project in your Drive by itself.

Progress is saved to your Drive as it goes, so if Colab disconnects, run the notebook again and it carries on where it stopped.

## What's in the folder

| File or folder | What it is |
|---|---|
| `nova/operator.py` | The heart of NOVA: the guess, cross-check and improve steps |
| `nova/model.py` | The complete NOVA model built around that core |
| `nova/baselines.py` | The models NOVA is compared against |
| `nova/tasks.py` | Makes the puzzles: number sequences, coloured grids and networks of connected points |
| `nova/splits.py` | Keeps practice puzzles and test puzzles strictly apart |
| `nova/train.py` | Teaches the models, saving progress safely as it goes |
| `nova/evaluate.py`, `nova/metrics.py` | Score the models and work out how confident to be in the scores |
| `nova/interface.py`, `nova/console.py` | A small app for sending NOVA a puzzle and seeing its answer and reasoning |
| `nova/tests/` | The 79 automatic checks |
| `tools/seed_bugs.py` | The planted-bug test |
| `NOVA_Research.ipynb` | The notebook that runs everything, step by step |
| `preregistration.md` | The plan, written down in advance |

## How the comparison is kept fair

- **Same size.** Every model NOVA is compared against has the same amount of learning capacity, within 5%.
- **Same effort.** One of the comparison models is given about the same amount of computing work as NOVA, so NOVA
  can't win just by doing more calculation.
- **Same practice.** Every model gets the same number of practice puzzles and training steps.
- **No peeking.** Test puzzles never appear in practice, and an automatic audit stops everything if any overlap is found.
- **A truly new test.** One set of puzzles uses a kind of rule no model ever practised on. It stays locked until the
  very end, can be opened only once, and every opening is recorded.
- **Honest records.** Every training run is logged, including the ones that crash or fail.

## Can the results be repeated?

Yes. Running the same experiment with the same settings gives exactly the same numbers. Saved progress is checked for
damage before it's used. Each result is stored with a fingerprint of the exact code and settings that produced it.



