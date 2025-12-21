import torch
from datasets import load_dataset
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
        
        # Get index fields based on dataset type
        if 'normalized_combined_ner' in source_dataset:  # VoxPopuli
            source_indices = source_dataset['id']
            index_type = 'voxpopuli'
        elif 'issue_id' in source_dataset:  # HVB
            source_issue_ids = source_dataset['issue_id']
            source_utt_indices = source_dataset['utt_index']
            index_type = 'hvb'
        else:  # VoxCeleb
            source_indices = source_dataset['index']
            index_type = 'voxceleb'
        
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
                    'label': label,  # This will be the filtered NER dict
                    'similarity_score': similarity_scores[idx][j]
                }
                
                # Add index information based on dataset type
                if index_type == 'voxpopuli':
                    example['index'] = source_indices[source_idx]
                elif index_type == 'hvb':
                    example['index'] = f"{source_issue_ids[source_idx]}_{source_utt_indices[source_idx]}"
                else:  # voxceleb
                    example['index'] = str(source_indices[source_idx])
                
                examples.append(example)
            few_shot_examples.append(examples)
        
        # Add few-shot examples to target dataset
        target_dataset = target_dataset.add_column('few_shot_examples', few_shot_examples)
        
        return target_dataset

def create_audio_lookup_dataset(datasets, subset, source_splits=["train", "validation"]):
    """Create a lookup dataset containing only audio and index information"""
    combined_audio = []
    combined_indices = []
    
    for split in source_splits:
        dataset = datasets[split]
        for item in dataset:
            if subset == "hvb":
                index = f"{item['issue_id']}_{item['utt_index']}"
            elif subset == "voxpopuli":  # VoxPopuli
                index = item['id']
            else:  # voxceleb
                index = str(item['index'])
            combined_indices.append(index)
            combined_audio.append(item['audio'])
    
    return {
        'audio': combined_audio,
        'index': combined_indices
    }

def main():
    # Choose dataset configuration
    DATASET_CONFIG = "voxpopuli"  # Options: "voxpopuli", "voxceleb", "hvb"
    
    if DATASET_CONFIG == "voxpopuli":
        text_column = 'normalized_text'
        label_column = 'normalized_combined_ner'  # or 'raw_ner'
        dataset_name = "asapp/slue"
        subset = "voxpopuli"
    elif DATASET_CONFIG == "voxceleb":
        text_column = 'normalized_text'
        label_column = 'sentiment'
        dataset_name = "asapp/slue"
        subset = "voxceleb"
    else:  # hvb
        text_column = 'text'
        label_column = 'dialog_acts'
        dataset_name = "asapp/slue-phase-2"
        subset = "hvb"

    source_split = ["train"]
    # source_split = ["validation"]
    target_split = "train"
    top_k = 5
    gpu_id = 1

    # Initialize few-shot generator
    generator = FewShotGenerator(gpu_id=gpu_id)
    
    # Load datasets
    datasets = generator.load_datasets(dataset_name, subset)

    # Combine source splits
    combined_texts = []
    combined_labels = []
    
    # Dataset-specific fields
    combined_ids = []
    combined_issue_ids = []
    combined_utt_indices = []
    combined_indices = []

    for split in source_split:
        combined_texts.extend(datasets[split][text_column])
        
        if DATASET_CONFIG == "voxpopuli":
            # Don't convert NER format here, just keep the original format
            combined_labels.extend(datasets[split][label_column])
            combined_ids.extend(datasets[split]['id'])
        else:
            combined_labels.extend(datasets[split][label_column])
            if DATASET_CONFIG == "hvb":
                combined_issue_ids.extend(datasets[split]['issue_id'])
                combined_utt_indices.extend(datasets[split]['utt_index'])
            elif DATASET_CONFIG == "voxceleb":
                combined_indices.extend(datasets[split]['index'])
    
    # Create the source dataset dictionary
    source_dataset = {
        f'{text_column}': combined_texts,
        f'{label_column}': combined_labels
    }
    
    # Add dataset-specific fields
    if DATASET_CONFIG == "voxpopuli":
        source_dataset['id'] = combined_ids
    elif DATASET_CONFIG == "hvb":
        source_dataset['issue_id'] = combined_issue_ids
        source_dataset['utt_index'] = combined_utt_indices
    elif DATASET_CONFIG == "voxceleb":
        source_dataset['index'] = combined_indices
    
    # Generate few-shot examples
    data_with_fewshots = generator.generate_fewshot_examples(
        target_dataset=datasets[target_split],
        source_dataset=source_dataset,
        top_k=top_k,
        text_column=text_column,
        label_column=label_column,
        is_source_in_target=(target_split in source_split)
    )
    # Example of how to access the few-shot examples
    print("Example few-shot data for first  instance:")
    print(data_with_fewshots['few_shot_examples'][0])
    
    # Save the augmented test dataset
    data_with_fewshots.save_to_disk(f"/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/{dataset_name}_{subset}_{target_split}_{top_k}fewshots")
    print(f"Dataset saved successfully to /home/leapers/weights/neeraja/ICL-speech-text-LLM/data/{dataset_name}_{subset}_{target_split}_{top_k}fewshots")

    # Create and save audio lookup dataset
    audio_lookup = create_audio_lookup_dataset(datasets, subset, source_splits=source_split)
    save_path = f"/home/leapers/weights/neeraja/ICL-speech-text-LLM/data/{dataset_name}_{subset}_{target_split}_audio_lookup"
    Dataset.from_dict(audio_lookup).save_to_disk(save_path)
    print(f"Audio lookup dataset saved to {save_path}")

if __name__ == "__main__":
    main()


