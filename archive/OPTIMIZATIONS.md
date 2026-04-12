# Performance Optimizations for ICL Framework

This document outlines the performance optimizations implemented in the ICL framework to improve training and inference speed.

## Training Optimizations

### Memory Efficiency

- **Gradient Checkpointing**: Reduces memory usage by not storing all intermediate activations. Enable with `--gradient_checkpointing`.
- **Mixed Precision Training**: Supports both FP16 (`--fp16`) and BF16 (`--bf16`) for faster computation with lower memory usage.
- **Memory Management**: Automatic garbage collection and CUDA cache clearing between validation runs.
- **Efficient Gradient Accumulation**: Uses `zero_grad(set_to_none=True)` for more efficient memory clearing.

### Data Loading

- **Dataset Caching**: Datasets are cached in memory to avoid repeated loading.
- **Parallel Processing**: Audio lookup tables are applied in parallel using ThreadPoolExecutor.
- **Optimized DataLoader**: Configurable number of workers, prefetch factor, and pin memory settings.
- **Persistent Workers**: Keeps worker processes alive between batches to reduce overhead.

### Computation Efficiency

- **Scheduler Options**: Multiple learning rate scheduler options (linear, cosine, polynomial, etc.).
- **Automatic Batch Size Optimization**: `BatchSizeOptimizer` class finds the optimal batch size for your hardware.
- **cuDNN Benchmarking**: Automatically enabled for operations with fixed input sizes.
- **Thread Optimization**: Configurable PyTorch thread settings for optimal CPU utilization.

### Model Efficiency

- **Model Caching**: Models are cached to avoid reloading the same model multiple times.
- **Gradient Clipping**: Prevents exploding gradients for more stable training.
- **Torch Compile**: Automatically applies `torch.compile()` for PyTorch 2.0+ when available.

## Monitoring and Debugging

- **Performance Tracking**: `PerformanceTracker` class for monitoring training speed, loss, and throughput.
- **Memory Usage Monitoring**: Utilities to track and log GPU and system memory usage.
- **Timing Utilities**: Decorators and context managers for timing code execution.
- **System Information Logging**: Comprehensive logging of hardware and software configuration.

## Usage Examples

### Enabling Mixed Precision Training

```bash
python train/train.py --model_path /path/to/model --fp16
```

For newer GPUs with Tensor Cores (Ampere, Ada Lovelace architectures):

```bash
python train/train.py --model_path /path/to/model --bf16
```

### Optimizing Memory Usage

```bash
python train/train.py --model_path /path/to/model --gradient_checkpointing --fp16
```

### Finding Optimal Batch Size

```python
from utils.performance_utils import BatchSizeOptimizer

# Create a function that returns a sample batch of given size
def get_sample_batch(batch_size):
    # Create a sample batch
    return {...}

# Initialize the optimizer
optimizer = BatchSizeOptimizer(
    model=model,
    sample_batch_fn=get_sample_batch,
    min_batch_size=1,
    max_batch_size=64
)

# Find the optimal batch size
optimal_batch_size = optimizer.find_optimal_batch_size()
print(f"Optimal batch size: {optimal_batch_size}")
```

### Tracking Performance

```python
from utils.performance_utils import PerformanceTracker

# Initialize the tracker
tracker = PerformanceTracker(log_interval=10)

# In your training loop
for batch in dataloader:
    start_time = time.time()
    
    # Forward and backward pass
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # Update tracker
    step_time = time.time() - start_time
    tracker.update(step_time, batch_size=len(batch), loss=loss.item())

# Get performance summary
tracker.log_summary()
```

### Optimizing PyTorch Settings

```python
from utils.performance_utils import optimize_torch_settings, log_system_info

# Apply optimal settings
optimizations = optimize_torch_settings()
print(f"Applied optimizations: {optimizations}")

# Log system information
log_system_info()
```

## Command Line Arguments

The training script now supports the following additional arguments:

- `--fp16`: Enable mixed precision training with FP16
- `--bf16`: Enable mixed precision training with BF16 (better for newer GPUs)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory
- `--dataloader_num_workers`: Number of workers for data loading
- `--dataloader_prefetch_factor`: Number of batches to prefetch
- `--scheduler`: Learning rate scheduler type (linear, cosine, etc.)
- `--seed`: Random seed for reproducibility
- `--eval_batch_size`: Separate batch size for evaluation

## Best Practices

1. **Always use mixed precision** (`--fp16` or `--bf16`) for faster training.
2. **Enable gradient checkpointing** for large models to reduce memory usage.
3. **Set appropriate number of workers** for your system (typically CPU count or slightly less).
4. **Monitor memory usage** during training to identify bottlenecks.
5. **Use the BatchSizeOptimizer** to find the optimal batch size for your hardware.
6. **Clear caches** between validation runs to free up memory.
7. **Use cosine scheduler** for better convergence on most tasks.

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size, enable gradient checkpointing, or use mixed precision.
- **Slow Data Loading**: Increase number of workers, enable pin memory, and check disk I/O.
- **NaN Losses**: Reduce learning rate, enable gradient clipping, or check for data issues.
- **Slow Training**: Enable mixed precision, optimize thread settings, and check GPU utilization.

## Future Improvements

- Implement DeepSpeed or FSDP for distributed training
- Add quantization support for inference
- Implement more advanced memory optimization techniques
- Add support for CPU offloading for very large models 