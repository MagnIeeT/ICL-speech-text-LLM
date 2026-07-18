from llava.train.train import train

if __name__ == "__main__":
    # Forced to eager (not flash_attention_2) so training and inference share the
    # same deterministic attention math — FlashAttention-2's tiled kernel reduction
    # order is not guaranteed identical across separate process/kernel-launch
    # contexts, which was producing bf16-ULP-scale logit drift between the live
    # training-validation forward pass and standalone inference on the same
    # checkpoint. See current_status.md Part 5/6.
    train(attn_implementation="eager")
