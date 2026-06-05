from llava.train.train import train

if __name__ == "__main__":
    try:
        import flash_attn  # noqa: F401
        train(attn_implementation="flash_attention_2")
    except ImportError:
        train(attn_implementation="eager")
