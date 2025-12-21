import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets
import logging
import json
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_sqa5_dataset():
    dataset_name = "asapp/slue-phase-2"
    subset = "sqa5"
    cache_dir = '/data2/neeraja/neeraja/data'
    
    try:
        for split in ['train','validation', 'test']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {split} split...")
            
            if split == 'train':
                split_dataset = load_dataset(
                    dataset_name, 
                    subset, 
                    split='train', 
                    cache_dir=cache_dir
                ).select(range(10000))
                logger.info(f"Selected first 10000 examples from train split")
            else:
                split_dataset = load_dataset(
                    dataset_name, 
                    subset, 
                    split=split, 
                    cache_dir=cache_dir
                )
            
            logger.info(f"Initial number of examples: {len(split_dataset)}")
            
            # Add unique_id column
            logger.info("\n=== Adding unique_id column ===")
            split_dataset = split_dataset.add_column("unique_id", 
                [f"{q}_{d}" for q, d in zip(split_dataset['question_id'], split_dataset['document_id'])]
            )
            
            # Add answer_text column
            logger.info("\n=== Adding answer_text column ===")
            split_dataset = split_dataset.add_column("answer_text",
                [json.loads(s)['answer'][0] if isinstance(s, str) else s['answer'][0] 
                 for s in split_dataset['answer_spans']]
            )
            
            # Add time_spans column
            logger.info("\n=== Adding time_spans column ===")
            split_dataset = split_dataset.add_column("time_spans",
                [[json.loads(s)['start_second'][0], json.loads(s)['end_second'][0]] if isinstance(s, str) else [s['start_second'][0], s['end_second'][0]] 
                 for s in split_dataset['answer_spans']]
            )
            
            # Log unique IDs stats
            num_examples = len(split_dataset)
            num_unique_ids = len(set(split_dataset['unique_id']))
            logger.info(f"Total examples: {num_examples}")
            logger.info(f"Unique IDs: {num_unique_ids}")
            if num_examples != num_unique_ids:
                logger.warning(f"Found {num_examples - num_unique_ids} duplicate IDs!")
            
            # Log answer stats
            answer_types = split_dataset['answer_text'][:5]
            logger.info("First 5 answers:")
            for i, ans in enumerate(answer_types):
                logger.info(f"Answer {i+1}: {ans} (type: {type(ans)})")
            
            # Log time spans stats
            time_spans = split_dataset['time_spans'][:5]
            logger.info("First 5 time spans:")
            for i, span in enumerate(time_spans):
                logger.info(f"Time span {i+1}: {span} (type: {type(span)})")
            
            # Final verification
            logger.info("\n=== Final Dataset Stats ===")
            logger.info(f"Number of examples: {len(split_dataset)}")
            logger.info(f"Columns: {split_dataset.column_names}")
            
            
            # Save processed dataset
            output_path = f"/data2/neeraja/neeraja/data/{dataset_name}_{subset}_{split}"
            split_dataset.save_to_disk(output_path)
            logger.info(f"\nSaved {split} split to {output_path}")
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

def process_vp_nel_dataset():
    dataset_name = "asapp/slue-phase-2"
    subset = "vp_nel"
    cache_dir = '/data2/neeraja/neeraja/data'
    
    try:
        logger.info(f"Loading {dataset_name} {subset} dataset...")
        
        for split in ['validation', 'test']:  # Only validation and test splits
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {split} split...")
            
            # Load split
            split_dataset = load_dataset(
                dataset_name, 
                subset, 
                split=split, 
                cache_dir=cache_dir
            )
            
            logger.info(f"Initial number of examples: {len(split_dataset)}")
            
            # Process ne_timestamps to create clean format
            logger.info("\n=== Processing NE timestamps ===")
            split_dataset = split_dataset.map(
                lambda x: {
                    'ne_spans': [
                        {
                            'label': label,
                            'time_span': [start, end]
                        }
                        for label, start, end in zip(
                            x['ne_timestamps']['ne_label'],
                            x['ne_timestamps']['start_sec'],
                            x['ne_timestamps']['end_sec']
                        )
                    ] if x['ne_timestamps'] else []
                },
                desc="Processing NE timestamps"
            )
            
            # Add unique_id column
            logger.info("\n=== Adding unique_id column ===")
            split_dataset = split_dataset.add_column(
                "unique_id", 
                [str(id_) for id_ in split_dataset['id']]
            )
            
            # Save processed dataset
            output_path = f"/data2/neeraja/neeraja/data/{dataset_name}_{subset}_{split}"
            split_dataset.save_to_disk(output_path)
            logger.info(f"Saved {split} split to {output_path}")
            
            # Log some stats
            logger.info(f"\nSplit {split} statistics:")
            logger.info(f"Number of examples: {len(split_dataset)}")
            
            # Show an example
            if len(split_dataset) > 0:
                example = split_dataset[0]
                logger.info("\nExample data point:")
                logger.info(f"ID: {example['id']}")
                logger.info(f"Text: {example['text'][:100]}...")
                logger.info(f"NE spans: {example['ne_spans']}")
                logger.info(f"Unique ID: {example['unique_id']}")
            
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

def process_meld_dataset():
    # cache_dir = '/data2/neeraja/neeraja/data'
    dataset_name = "zrr1999/MELD_Text_Audio"
    
    emotion_labels = {
        0: "neutral",
        1: "joy",
        2: "sadness",
        3: "anger", 
        4: "fear",
        5: "disgust",
        6: "surprise"
    }
    
    # Sentiment mapping 
    sentiment_labels = {
        0: "neutral",
        1: "positive",
        2: "negative"
    }
    
    try:
        # Process each split
        for split in ['train']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {split} split...")
            
            split_dataset = load_dataset(
                dataset_name,
                split=split
            )
            
            logger.info(f"Initial number of examples: {len(split_dataset)}")
            
            # Filter out examples with missing audio files
            logger.info("\n=== Filtering examples with missing audio files ===")
            valid_indices = []
            for i, path in enumerate(split_dataset['path']):
                if os.path.exists(path):
                    valid_indices.append(i)
                else:
                    logger.warning(f"Audio file not found: {path}")
            
            split_dataset = split_dataset.select(valid_indices)
            logger.info(f"Remaining examples after filtering: {len(split_dataset)}")
            
            # Add unique_id column based on audio path
            logger.info("\n=== Adding unique_id column ===")
            split_dataset = split_dataset.add_column(
                "unique_id", 
                [f"meld_{os.path.basename(p).replace('.flac', '')}" for p in split_dataset['path']]
            )
            
            # Convert emotion integer to text label
            logger.info("\n=== Converting emotion to text labels ===")
            split_dataset = split_dataset.add_column(
                "emotion_label",
                [emotion_labels.get(e, "unknown") for e in split_dataset['emotion']]
            )
            
            # Convert sentiment integer to text label
            logger.info("\n=== Converting sentiment to text labels ===")
            split_dataset = split_dataset.add_column(
                "sentiment_label",
                [sentiment_labels.get(s, "unknown") for s in split_dataset['sentiment']]
            )
            
            # # Create audio lookup
            # logger.info("\n=== Creating audio lookup ===")
            # audio_lookup = {uid: path for uid, path in zip(split_dataset['unique_id'], split_dataset['path'])}
            
            # Save processed dataset
            output_path = f"/data2/neeraja/neeraja/data/meld_{split}"
            split_dataset.save_to_disk(output_path)
            logger.info(f"Saved {split} split to {output_path}")
            
            # # Save audio lookup
            # audio_lookup_path = f"/data2/neeraja/neeraja/data/meld_{split}_audio_lookup"
            # os.makedirs(os.path.dirname(audio_lookup_path), exist_ok=True)
            # with open(f"{audio_lookup_path}.json", 'w') as f:
            #     json.dump(audio_lookup, f)
            # logger.info(f"Saved audio lookup to {audio_lookup_path}.json")
            
            # Log some stats
            logger.info(f"\nSplit {split} statistics:")
            logger.info(f"Number of examples: {len(split_dataset)}")
            
            # Show an example
            if len(split_dataset) > 0:
                example = split_dataset[0]
                logger.info("\nExample data point:")
                logger.info(f"Text: {example['text'][:100]}...")
                logger.info(f"Emotion: {example['emotion']} → {example['emotion_label']}")
                logger.info(f"Sentiment: {example['sentiment']} → {example['sentiment_label']}")
                logger.info(f"Unique ID: {example['unique_id']}")
            
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # process_sqa5_dataset()
    # process_vp_nel_dataset()
    process_meld_dataset()