# Sequential Implementation (Baseline)

This folder contains the original sequential implementations using Array of Structures (AoS) approach.

## Files

- `Water_sequential.py` - Main sequential implementation with object-oriented design
- `Water_sequential_lists.py` - Alternative sequential implementation using Python lists

## Key Characteristics

- **Memory Layout**: Array of Structures (AoS)
- **Language**: Pure Python with minimal NumPy
- **Approach**: Object-oriented with individual Molecule objects
- **Performance**: Baseline performance (1x speedup)
- **Complexity**: O(N²) for distance calculations

## Usage

```python
python Water_sequential.py
```

## Performance Profile

- Distance calculations: 65.2% - 84.3% of runtime (scales with system size)
- Force calculations: ~8% of runtime  
- Integration: ~3% of runtime
- Memory usage: Highest due to Python object overhead

This implementation serves as the baseline for all performance comparisons.