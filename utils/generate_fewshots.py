import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.cuda
import logging
from datasets import Dataset

def convert_ner_to_dict(text: str, ner_data: Dict) -> Dict[str, List[str]]:
    """
    Convert NER data from start/length format to {tag: [phrases]} format.
    Only includes tags that have actual phrases (not None).
    
    Args:
        text: The input text
        ner_data: Dictionary containing tag-phrase mappings
    
    Returns:
        Dictionary mapping entity types to lists of phrases, excluding empty tags
    """
    result = {}
    
    # For the original start/length format
    for tag, start, length in zip(ner_data['type'], ner_data['start'], ner_data['length']):
        # Extract the phrase using start and length
        phrase = text[start:start + length]
        
        # Only add non-empty phrases
        if phrase.strip():
            if tag not in result:
                result[tag] = []
            result[tag].append(phrase)
    
    return result

class FewShotGenerator:
    def __init__(self, model_name: str = 'BAAI/llm-embedder', gpu_id: int = 1):
        """
        Initialize the FewShotGenerator
        Args:
            model_name: Name of the model to use for embeddings
            cache_dir: Directory to cache the datasets
            gpu_id: GPU device ID to use (e.g., 0, 1, 2, etc.)
        """
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"Available GPUs: {torch.cuda.device_count()}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device("cpu")
            print("No GPU available, using CPU")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def load_datasets(self, dataset_name: str, subset: str) -> Dict:
        """Load train, validation and test splits of a dataset"""
        return {
            'train': load_dataset(dataset_name, subset, split="train"),
            'validation': load_dataset(dataset_name, subset, split="validation"),
            'test': load_dataset(dataset_name, subset, split="test")
        }

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for a list of texts"""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # Move input tensors to GPU
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            # Use CLS token embedding
            embeddings = outputs[0][:, 0]
            # Move embeddings back to CPU for sklearn compatibility
            embeddings = embeddings.cpu()
        return embeddings

    def find_similar_examples(self, 
                            query_texts: List[str],
                            source_texts: List[str],
                            top_k: int = 5,
                            batch_size: int = 32,
                            is_source_in_target: bool = False) -> Tuple[List[List[int]], List[List[float]]]:
        """Find top-k similar examples from source texts for each query text"""
        
        # Process query texts in batches
        query_embeddings_list = []
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch_texts)
            query_embeddings_list.append(batch_embeddings)
        query_embeddings = torch.cat(query_embeddings_list, dim=0)

        # Process source texts in batches
        source_embeddings_list = []
        for i in range(0, len(source_texts), batch_size):
            batch_texts = source_texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch_texts)
            source_embeddings_list.append(batch_embeddings)
        source_embeddings = torch.cat(source_embeddings_list, dim=0)
        
        # Normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        source_embeddings = torch.nn.functional.normalize(source_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embeddings.numpy(), source_embeddings.numpy())
        
        # Get top-k indices and scores
        top_k_indices = []
        top_k_scores = []
        for sim_row in similarities:
            # Only filter out perfect matches if source split is in target split
            if is_source_in_target:
                valid_mask = sim_row < 1.0
                filtered_scores = sim_row[valid_mask]
                filtered_indices = np.arange(len(sim_row))[valid_mask]
            else:
                filtered_scores = sim_row
                filtered_indices = np.arange(len(sim_row))
            
            # Create a dictionary to store unique texts with their highest similarity scores
            unique_texts = {}
            for idx, score in zip(filtered_indices, filtered_scores):
                text = source_texts[idx]
                # Keep the highest similarity score for each unique text
                if text not in unique_texts or score > unique_texts[text]['score']:
                    unique_texts[text] = {
                        'index': idx,
                        'score': score
                    }
            
            # Convert back to arrays for sorting
            unique_indices = np.array([info['index'] for info in unique_texts.values()])
            unique_scores = np.array([info['score'] for info in unique_texts.values()])
            
            # Get top-k from unique results
            k_idx = min(top_k, len(unique_scores))  # In case we have fewer unique examples than top_k
            top_indices = unique_indices[np.argsort(unique_scores)[-k_idx:][::-1]]
            top_scores = unique_scores[np.argsort(unique_scores)[-k_idx:][::-1]]
            
            top_k_indices.append(top_indices.tolist())
            top_k_scores.append(top_scores.tolist())
            
        return top_k_indices, top_k_scores

    def generate_fewshot_examples(self,
                                target_dataset: Dict,
                                source_dataset: Dict,
                                text_column: str = 'normalized_text',
                                label_column: str = 'sentiment',
                                top_k: int = 5,
                                is_source_in_target: bool = False) -> Dict:
        """Generate few-shot examples for target dataset using source dataset"""
        
        # Extract texts and labels from source dataset
        source_texts = source_dataset[text_column]
        source_labels = source_dataset[label_column]
        
        # ✅ Fix: Use the standardized 'id' key provided by the main function
        source_indices = source_dataset['id']
        
        # Find similar examples
        similar_indices, similarity_scores = self.find_similar_examples(
            target_dataset[text_column],
            source_texts,
            top_k=top_k,
            is_source_in_target=is_source_in_target
        )
        
        # Format few-shot examples
        few_shot_examples = []
        for idx, indices in enumerate(similar_indices):
            examples = []
            for j, source_idx in enumerate(indices):
                # Convert NER format if needed
                if label_column == 'normalized_combined_ner':
                    label = convert_ner_to_dict(source_texts[source_idx], source_labels[source_idx])
                else:
                    label = source_labels[source_idx]
                
                example = {
                    'text': source_texts[source_idx],
                    'label': label,
                    'similarity_score': similarity_scores[idx][j],
                    'index': str(source_indices[source_idx]) # ✅ Use standardized index
                }
                
                examples.append(example)
            few_shot_examples.append(examples)
        
        # Add few-shot examples to target dataset
        target_dataset = target_dataset.add_column('few_shot_examples', few_shot_examples)
        
        return target_dataset

def create_audio_lookup_dataset(datasets, dataset_config, source_splits=["train"]):
    """Create a lookup dataset containing only audio and index information"""
    combined_audio = []
    combined_indices = []
    
    for split in source_splits:
        dataset = datasets[split]
        for item in dataset:
            # ✅ Updated to handle all 5 datasets
            if dataset_config == "hvb":
                index = f"{item['issue_id']}_{item['utt_index']}"
            elif dataset_config == "voxpopuli":
                index = item['id']
            elif dataset_config in ["meld", "vp_nel"]:
                index = item['unique_id']
            else:  # voxceleb
                index = str(item['index'])
                
            combined_indices.append(index)
            combined_audio.append(item['audio'])
    
    return {
        'audio': combined_audio,
        'index': combined_indices
    }

from .environment import get_env_path

def main():
    # Choose dataset configuration
    # Options: "voxpopuli", "voxceleb", "hvb", "meld", "vp_nel"
    DATASET_CONFIG = "voxceleb" 
    
    # Define splits
    target_split = "test"      # Split to generate few-shots FOR
    source_splits = ["train"]  # Split to pick examples FROM
    top_k = 50
    
    PROCESSED_BASE_PATH = get_env_path("BASE_DATA_DIR")
    # 1. Configure Dataset Metadata
    if DATASET_CONFIG == "voxpopuli":
        text_column, label_column = 'normalized_text', 'normalized_combined_ner'
        dataset_name, subset = "asapp/slue", "voxpopuli"
        use_processed = False
    elif DATASET_CONFIG == "voxceleb":
        text_column, label_column = 'normalized_text', 'sentiment'
        dataset_name, subset = "asapp/slue", "voxceleb"
        use_processed = False
    elif DATASET_CONFIG == "meld":
        text_column, label_column = 'text', 'emotion_label'
        use_processed = True
    elif DATASET_CONFIG == "vp_nel":
        text_column, label_column = 'text', 'ne_spans'
        use_processed = True
    else: # hvb
        text_column, label_column = 'text', 'dialog_acts'
        dataset_name, subset = "asapp/slue-phase-2", "hvb"
        use_processed = False

    # 2. Load Datasets
    datasets = {}
    all_needed_splits = list(set([target_split] + source_splits))
    generator = FewShotGenerator(gpu_id=1)

    for split in all_needed_splits:
        if use_processed:
            path = f"{PROCESSED_BASE_PATH}/{DATASET_CONFIG}_{split}"
            logging.info(f"Loading processed {split} split from: {path}")
            datasets[split] = load_from_disk(path)
        else:
            datasets[split] = load_dataset(dataset_name, subset, split=split)

    # 3. Combine Source Splits for Example Pool
    combined_texts, combined_labels, combined_ids = [], [], []
    for split in source_splits:
        ds = datasets[split]
        combined_texts.extend(ds[text_column])
        combined_labels.extend(ds[label_column])
        
        if use_processed:
            combined_ids.extend(ds['unique_id'])
        elif DATASET_CONFIG == "voxpopuli":
            combined_ids.extend(ds['id'])
        elif DATASET_CONFIG == "hvb":
            combined_ids.extend([f"{i}_{u}" for i, u in zip(ds['issue_id'], ds['utt_index'])])
        else: # voxceleb
            combined_ids.extend([str(i) for i in ds['index']])
    
    source_dataset = {
        text_column: combined_texts,
        label_column: combined_labels,
        'id': combined_ids
    }
    
    # 4. Generate Few-Shots
    data_with_fewshots = generator.generate_fewshot_examples(
        target_dataset=datasets[target_split],
        source_dataset=source_dataset,
        top_k=top_k,
        text_column=text_column,
        label_column=label_column,
        is_source_in_target=(target_split in source_splits)
    )

    # 5. Save Augmented Dataset
    save_name = f"{DATASET_CONFIG}_{target_split}_{top_k}fewshots"
    data_with_fewshots.save_to_disk(f"{PROCESSED_BASE_PATH}/{save_name}")
    print(f"Dataset saved to {PROCESSED_BASE_PATH}/{save_name}")

    # 6. Create and Save Audio Lookup
    audio_lookup = create_audio_lookup_dataset(datasets, DATASET_CONFIG, source_splits=source_splits)
    lookup_save_path = f"{PROCESSED_BASE_PATH}/{DATASET_CONFIG}_{target_split}_audio_lookup"
    Dataset.from_dict(audio_lookup).save_to_disk(lookup_save_path)
    print(f"Audio lookup saved to {lookup_save_path}")

if __name__ == "__main__":
    main()


