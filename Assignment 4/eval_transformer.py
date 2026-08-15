"""
eval.py - Evaluation script for Maze Solver models (RNN/Transformer)

Usage:
    python eval.py <model_path> <model_type> <data_csv_path> <output_csv_path>

Arguments:
    model_path: Path to the pre-trained model (.pt file)
    model_type: Type of model - 'rnn' or 'transformer' (lowercase)
    data_csv_path: Path to input CSV file with test data
    output_csv_path: Path to save output CSV file

The script automatically loads vocabulary from 'vocab.pkl' in the current directory.

Example:
    python eval.py ./best_transformer_model.pt transformer ./test.csv ./submission.csv
"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import ast
import math
import pickle
from torch.utils.data import Dataset

# ============================================================================
# VOCABULARY CLASS
# ============================================================================

class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<>'  # Matches data format
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'

        # Add special tokens
        self.add_token(self.PAD_TOKEN)
        self.add_token(self.SOS_TOKEN)
        self.add_token(self.EOS_TOKEN)
        self.add_token(self.UNK_TOKEN)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_from_sequences(self, sequences):
        for seq in sequences:
            for token in seq:
                self.add_token(token)

    def encode(self, tokens):
        return [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        return [self.idx2token[i] for i in indices if i in self.idx2token]

    def __len__(self):
        return len(self.token2idx)


# ============================================================================
# DATASET CLASS
# ============================================================================

class MazeDataset(Dataset):
    def __init__(self, csv_path, vocab=None, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        # Parse sequences
        self.inputs = [ast.literal_eval(s) for s in self.df['input_sequence']]
        
        # Build or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_from_sequences(self.inputs)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        input_encoded = self.vocab.encode(input_seq)

        # Handle both cases: with or without 'id' column
        if 'id' in self.df.columns:
            sample_id = self.df.iloc[idx]['id']
        else:
            sample_id = idx + 1  # Use 1-indexed IDs if no 'id' column

        return {
            'input': torch.tensor(input_encoded, dtype=torch.long),
            'input_len': len(input_encoded),
            'id': sample_id
        }


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class TransformerMazeSolver(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, 
                 dim_feedforward=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        with torch.no_grad():
            self.embedding.weight[0] = 0

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx=0):
        return (seq == pad_idx)

    def forward(self, src, tgt):
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        output = self.fc_out(output)
        return output

    def predict(self, src, max_len=50, start_token=1, end_token=2):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)
            predictions = []

            for i in range(max_len):
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
                tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)

                output = self.transformer_decoder(
                    tgt_emb, memory, tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                output = self.fc_out(output)
                next_token = output[:, -1, :].argmax(dim=-1)
                predictions.append(next_token)

                if (next_token == end_token).all():
                    break

                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            if predictions:
                predictions = torch.stack(predictions, dim=1)
            else:
                predictions = torch.zeros(batch_size, 0, dtype=torch.long).to(self.device)

            return predictions


# ============================================================================
# RNN MODEL (PLACEHOLDER - Add your RNN implementation if needed)
# ============================================================================

class RNNMazeSolver(nn.Module):
    """RNN-based maze solver - Add your implementation"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def predict(self, src, max_len=50, start_token=1, end_token=2):
        """Prediction method for RNN"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            
            # Encode
            src_emb = self.embedding(src)
            _, (hidden, cell) = self.encoder(src_emb)
            
            # Decode
            decoder_input = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)
            predictions = []
            
            for _ in range(max_len):
                decoder_emb = self.embedding(decoder_input)
                output, (hidden, cell) = self.decoder(decoder_emb, (hidden, cell))
                output = self.fc_out(output)
                
                next_token = output[:, -1, :].argmax(dim=-1)
                predictions.append(next_token)
                
                if (next_token == end_token).all():
                    break
                
                decoder_input = next_token.unsqueeze(1)
            
            if predictions:
                predictions = torch.stack(predictions, dim=1)
            else:
                predictions = torch.zeros(batch_size, 0, dtype=torch.long).to(self.device)
            
            return predictions


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model_path, model_type, data_csv_path, output_csv_path):
    """
    Evaluate model and generate predictions
    
    Args:
        model_path: Path to pre-trained model (.pt file)
        model_type: 'rnn' or 'transformer'
        data_csv_path: Path to input CSV
        output_csv_path: Path to save output CSV
    
    Note: Vocabulary is automatically loaded from 'vocab.pkl' in current directory
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary from vocab.pkl in current directory
    vocab_path = 'vocab_transformer.pkl'
    print(f"Loading vocabulary from: {vocab_path}")
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"✓ Vocabulary loaded successfully")
        print(f"  Vocabulary size: {len(vocab)}")
    except FileNotFoundError:
        print(f"ERROR: Vocabulary file not found: {vocab_path}")
        print("Please ensure 'vocab.pkl' is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load vocabulary: {e}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {data_csv_path}")
    df = pd.read_csv(data_csv_path)
    print(f"Loaded {len(df)} samples")
    
    # ========== SAMPLING FOR TESTING (Comment out for full evaluation) ==========
    df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)
    print(f"⚠ SAMPLING ENABLED: Using {len(df)} samples (10% of data)")
    # ============================================================================
    
    # Save sampled dataframe temporarily for dataset creation
    temp_csv_path = data_csv_path.replace('.csv', '_temp_sampled.csv')
    df.to_csv(temp_csv_path, index=False)
    
    # Create dataset
    dataset = MazeDataset(temp_csv_path, vocab=vocab, is_train=False)
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    
    # Use vocabulary size from the loaded vocabulary (from training)
    vocab_size = len(vocab)
    
    if model_type == 'transformer':
        model = TransformerMazeSolver(
            vocab_size=vocab_size,  # Use exact vocab size from training
            d_model=128,
            nhead=8,
            num_layers=6,
            dim_feedforward=512,
            dropout=0.1,
            device=device
        ).to(device)
    elif model_type == 'rnn':
        model = RNNMazeSolver(
            vocab_size=vocab_size,  # Use exact vocab size from training
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            device=device
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'rnn' or 'transformer'")
    
    # Load model weights
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Get special tokens
    start_token = vocab.token2idx.get('<PATH_START>', vocab.token2idx.get('<>', 1))
    end_token = vocab.token2idx.get('<PATH_END>', vocab.token2idx.get('<EOS>', 2))
    
    print(f"Start token: {start_token} ({vocab.idx2token[start_token]})")
    print(f"End token: {end_token} ({vocab.idx2token[end_token]})")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_list = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            src = sample['input'].unsqueeze(0).to(device)
            sample_id = sample['id']
            
            # Get original input and maze type from dataframe
            input_sequence = df.iloc[i]['input_sequence']
            maze_type = df.iloc[i]['maze_type'] if 'maze_type' in df.columns else 'unknown'
            
            # Pad to reasonable length for batch processing
            if src.size(1) < 256:
                padding = torch.zeros(1, 256 - src.size(1), dtype=torch.long, device=device)
                src = torch.cat([src, padding], dim=1)
            
            # Predict
            predictions = model.predict(src, max_len=50, start_token=start_token, end_token=end_token)
            
            # Decode
            predicted_indices = predictions[0].cpu().tolist()
            predicted_tokens = vocab.decode(predicted_indices)
            
            # Remove special tokens and padding
            predicted_tokens = [t for t in predicted_tokens 
                              if t not in ['<PAD>', '<>', '<EOS>', '<PATH_START>', '<UNK>']]
            
            # Manually add <PATH_END> token at the end
            predicted_tokens.append('<PATH_END>')
            
            # Store prediction with all required columns
            predictions_list.append({
                'id': f'id_{sample_id-1}',
                'input_path': input_sequence,
                'maze_type': maze_type,
                'output_path': str(predicted_tokens)
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples")
    
    print(f"✓ Completed predictions for {len(predictions_list)} samples")
    
    # Create output DataFrame with correct column order
    output_df = pd.DataFrame(predictions_list)
    
    # Ensure correct column order: id, input_path, maze_type, output_path
    required_columns = ['id', 'input_path', 'maze_type', 'output_path']
    if not all(col in output_df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in output_df.columns]
        raise ValueError(f"Output missing required columns: {missing}")
    
    # Reorder columns to match expected format
    output_df = output_df[required_columns]
    
    # Save predictions
    print(f"\nSaving predictions to: {output_csv_path}")
    output_df.to_csv(output_csv_path, index=False)
    print("✓ Predictions saved successfully!")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(output_df.head())
    
    # Clean up temporary file
    import os
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("="*70)
        print("ERROR: Incorrect number of arguments")
        print("="*70)
        print("\nUsage:")
        print("  python eval.py <model_path> <model_type> <data_csv_path> <output_csv_path>")
        print("\nArguments:")
        print("  model_path      : Path to pre-trained model (.pt file)")
        print("  model_type      : 'rnn' or 'transformer' (lowercase)")
        print("  data_csv_path   : Path to input CSV file")
        print("  output_csv_path : Path to save output CSV file")
        print("\nNote:")
        print("  Vocabulary is automatically loaded from 'vocab.pkl' in current directory")
        print("\nExample:")
        print("  python eval.py ./best_transformer.pt transformer ./test.csv ./output.csv")
        print("="*70)
        sys.exit(1)
    
    # Parse command line arguments
    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    data_csv_path = sys.argv[3]
    output_csv_path = sys.argv[4]
    
    # Print configuration
    print("="*70)
    print("MAZE SOLVER EVALUATION")
    print("="*70)
    print(f"Model Path      : {model_path}")
    print(f"Model Type      : {model_type}")
    print(f"Data CSV        : {data_csv_path}")
    print(f"Output CSV      : {output_csv_path}")
    print(f"Vocabulary      : vocab.pkl (from current directory)")
    print("="*70)
    print()
    
    # Run evaluation
    try:
        evaluate_model(model_path, model_type, data_csv_path, output_csv_path)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)