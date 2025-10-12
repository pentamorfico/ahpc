# Week 2 — Python task-farm assignment

This README summarizes the Week 2 assignment (see Assignment 2.pdf) and focuses on the Python implementation under `week2/python/`.

Goals
- Implement and benchmark a task-farm style parallelization of a HEP-like job in Python.
- Provide sequential and parallel versions to compare scaling and performance.

Repository files (Python)
- `task_farm.py` — final parallel implementation (expected to use multiprocessing or concurrent.futures).
- `task_farm_skeleton.py` — skeleton / starter code for students to complete.
- `task_farm_HEP.py` — full HEP-style task farm reference or example.
- `task_farm_HEP_seq.py` — sequential reference implementation.
- `job.sh` — example job script used by the HPC environment to run the Python implementation.

What you need to do (concrete tasks)
1. Read the assignment description in `Assignment 2.pdf` to understand the input data format and evaluation metrics.
2. Verify `task_farm_HEP_seq.py` runs correctly as a sequential baseline. Fix any obvious issues.
3. Implement a parallel task-farm in `task_farm.py`:
   - Use `multiprocessing.Pool`, `concurrent.futures.ProcessPoolExecutor`, or a work-stealing approach.
   - Ensure tasks are distributed dynamically (not static slicing) so long-running tasks don't cause load imbalance.
   - Keep the I/O minimal in worker processes; return small result objects to the main process.
4. Provide a lightweight CLI for `task_farm.py` and `task_farm_HEP.py`:
   - Arguments: input file path, number of workers, output path (optional), and a `--profile` flag to enable timing.
   - Print a short summary of runtime and per-task statistics at exit.
5. Ensure reproducible runs:
   - Add deterministic RNG seeding when required by the workload.
   - Make the batch size or task chunk size configurable.
6. Add a simple benchmark harness in `job.sh` or `bench.sh`:
   - Show runs for 1, 2, 4, 8 workers (if available on your machine).
   - Collect and append runtimes to `results/` or `results/python_runtime.txt`.
7. Add basic validation and unit tests:
   - Small tests for correctness (e.g., compare `task_farm.py` output to `task_farm_HEP_seq.py` on a small input).
   - A smoke test script `tests/run_smoke.sh` that runs sequential and parallel versions and compares outputs.

Suggested implementation notes and tips
- Use `if __name__ == '__main__':` guards to avoid fork issues on some platforms.
- Keep worker functions top-level (picklable) to work with `multiprocessing`.
- When returning results, prefer compact structures (tuples) and aggregate in the parent process.
- For large inputs, stream tasks from the input file rather than loading all tasks into memory at once.
- If using `concurrent.futures`, consider `as_completed()` to process results as they finish.
- Profile with `time.perf_counter()` or the `time` command in `job.sh` and compare speedups.

Example CLI usage

python3 task_farm.py --input data/sample.csv --workers 4 --output results/out.json --profile

Try-it commands (run in project root)

# run sequential baseline
python3 week2/python/task_farm_HEP_seq.py --input week2/mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv

# run parallel with 4 workers
python3 week2/python/task_farm.py --input week2/mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv --workers 4 --profile

Testing and validation
- Create a small synthetic input (e.g., 100 short tasks) and run both sequential and parallel versions to verify outputs match.
- Use assertions in tests to compare output checksums or result summaries.
- Optionally, use `pytest` for lightweight unit tests.

Deliverables for submission
- Completed and documented Python implementations in `week2/python/`.
- A working `job.sh` showing how to launch the parallel Python job in the target environment.
- A short report or `results/` entries showing runtimes for multiple worker counts and brief analysis.

Next steps (nice-to-have)
- Add a `benchmarks/` script that produces plots (matplotlib) of runtime vs workers.
- Add a small CI job to run smoke tests on pushes (GitHub Actions).
- Add type hints and linting configuration (mypy, flake8) for code quality.

If you want, I can:
- Implement the CLI skeleton in `task_farm.py` and add a smoke test and bench script.
- Run the smoke tests here (if you want me to run them, tell me how many workers to test with).

---
README generated automatically by assistant to summarize the assignment and list actionable tasks for the Python implementation.