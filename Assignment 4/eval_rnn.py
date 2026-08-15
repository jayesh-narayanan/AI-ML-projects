import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import sys
import os
import pickle
from tqdm import tqdm
import math
# --- 1. Common Vocabulary Class  ---
class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<>'
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
            self.idx2token[idx] = token
            self.token2idx[token] = idx

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

# --- 2. Dataset Class ---
class MazeDatasetRNN(Dataset):
    def __init__(self, csv_path, vocab=None):
        self.df = pd.read_csv(csv_path)
        
        # Parse inputs
        self.inputs = [ast.literal_eval(s) for s in self.df['input_sequence']]
        
        # Handle outputs (might be empty or missing in test files)
        if 'output_path' in self.df.columns:
            self.outputs = []
            for s in self.df['output_path']:
                if pd.isna(s) or (isinstance(s, str) and s.strip() == ''):
                    self.outputs.append([])
                else:
                    try:
                        self.outputs.append(ast.literal_eval(s))
                    except:
                        self.outputs.append([])
        else:
            self.outputs = [[] for _ in range(len(self.inputs))]

        # Vocab is mandatory now
        if vocab is None:
            raise ValueError("Vocabulary must be provided (loaded from vocab_rnn.pkl).")
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        input_encoded = self.vocab.encode(input_seq)
        
        # Encode output if available (just for consistency, not used for prediction)
        output_seq = self.outputs[idx] if idx < len(self.outputs) else []
        output_encoded = self.vocab.encode(output_seq)

        return {
            'input': torch.tensor(input_encoded, dtype=torch.long),
            'output': torch.tensor(output_encoded, dtype=torch.long),
            'input_len': len(input_encoded),
            'output_len': len(output_encoded)
        }

def collate_fn_rnn(batch):
    inputs = [item['input'] for item in batch]
    input_lens = [item['input_len'] for item in batch]
    output_lens = [item['output_len'] for item in batch]

    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    
    outputs = [item['output'] for item in batch if len(item['output']) > 0]
    if outputs:
        outputs_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)
    else:
        outputs_padded = None

    return {
        'input': inputs_padded,
        'output': outputs_padded,
        'input_len': torch.tensor(input_lens),
        'output_len': torch.tensor(output_lens)
    }

# --- 3. Model Classes ---
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        score = self.V(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))
        score = score.squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask, -1e9)
        attention_weights = F.softmax(score, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attention_weights

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(x))
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden, attention_weights

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, lengths, max_len):
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len) >= lengths.to(self.device).unsqueeze(1)
        return mask

    def predict(self, src, src_len, max_len=50, start_token=None, end_token=None):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            encoder_outputs, hidden = self.encoder(src, src_len)
            mask = self.create_mask(src_len, src.size(1))

            if start_token is None:
                start_token = 1
            decoder_input = torch.full((batch_size, 1), start_token, dtype=torch.long).to(self.device)
            predictions = []

            for _ in range(max_len):
                output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                predictions.append(top1)
                if end_token is not None and (top1 == end_token).all():
                    break
                decoder_input = top1.unsqueeze(1)

            return torch.stack(predictions, dim=1)

# --- 4. Functions ---

def load_rnn_model(model_path, vocab_size, device):
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.0
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle Checkpoint Wrapper (if user passes checkpoint instead of weights)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    encoder = EncoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = DecoderRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    model = Seq2SeqRNN(encoder, decoder, device).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_predictions_rnn(model, dataloader, vocab, device):
    model.eval()
    all_predictions = []
    
    # Automatically find start/end tokens from loaded vocab
    # Defaults to ID 1 and 2 if not found (standard for notebook)
    start_token = vocab.token2idx.get('<PATH_START>', vocab.token2idx.get('<>', 1))
    end_token = vocab.token2idx.get('<PATH_END>', vocab.token2idx.get('<EOS>', 2))
    
    print(f"Using Start Token: {start_token}, End Token: {end_token}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            src = batch['input'].to(device)
            src_len = batch['input_len'].to(device)
            
            predictions = model.predict(src, src_len, max_len=50,
                                       start_token=start_token,
                                       end_token=end_token)
            
            for i in range(predictions.size(0)):
                pred_indices = predictions[i].cpu().tolist()
                
                # Include end token in output for consistency with training
                try:
                    first_end_idx = pred_indices.index(end_token)
                    pred_indices = pred_indices[:first_end_idx + 1]
                except ValueError:
                    pass
                
                pred_tokens = vocab.decode(pred_indices)
                all_predictions.append(pred_tokens)
    
    return all_predictions

# --- Classes for Transformer --- 

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
# POSITIONAL ENCODING FOR TRANSFORMER
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
def evaluate_transformer_model(model_path, model_type, data_csv_path, output_csv_path):
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
    

    
    # Save sampled dataframe temporarily for dataset creation
    temp_csv_path = data_csv_path.replace('.csv', '_temp_sampled.csv')
    df.to_csv(temp_csv_path, index=False)
    
    # Create dataset
    dataset = MazeDataset(temp_csv_path, vocab=vocab, is_train=False)
    
    # Initialize model
    print(f"Initializing {model_type} model...")
    
    # Use vocabulary size from the loaded vocabulary (from training)
    vocab_size = len(vocab)
    

    model = TransformerMazeSolver(
            vocab_size=vocab_size,  # Use exact vocab size from training
            d_model=128,
            nhead=8,
            num_layers=6,
            dim_feedforward=512,
            dropout=0.1,
            device=device
        ).to(device)


    
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

def main():
    if len(sys.argv) != 5:
        print("Usage: python eval.py <model_path> <model_type> <input_csv_path> <output_csv_path>")
        print("Example: python eval.py best_model.pt rnn test.csv submission.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    input_csv_path = sys.argv[3]
    output_csv_path = sys.argv[4]

    if model_type=='rnn':
        # Fixed path for vocabulary
        vocab_path = "vocab_rnn.pkl"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 1. Load Vocabulary (FIXED: Must exist)
        if not os.path.exists(vocab_path):
            print(f"Error: '{vocab_path}' not found in current directory.")
            print("Please download vocab_rnn.pkl from your notebook and place it here.")
            sys.exit(1)
            
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        vocab_size = len(vocab)
        print(f"Loaded Vocabulary size: {vocab_size}")

        # 2. Load Data
        test_dataset = MazeDatasetRNN(input_csv_path, vocab=vocab)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_rnn)
        print(f"Number of test samples: {len(test_dataset)}")

        # 3. Load Model
        model = load_rnn_model(model_path, vocab_size, device)

        # 4. Generate
        print("Generating predictions...")
        predictions = generate_predictions_rnn(model, test_loader, vocab, device)

        # 5. Save
        output_df = pd.read_csv(input_csv_path)
        output_df['output_path'] = [str(pred) for pred in predictions]
        
        if 'id' not in output_df.columns:
            output_df.insert(0, 'id', [f'id_{i}' for i in range(len(output_df))])
        
        final_cols = [c for c in ['id', 'input_sequence', 'maze_type', 'output_path'] if c in output_df.columns]
        output_df = output_df[final_cols]
        
        output_df.to_csv(output_csv_path, index=False)
        print(f"Output saved to: {output_csv_path}")

    elif model_type=='transformer':
         # Run evaluation
        try:
            evaluate_transformer_model(model_path, model_type, input_csv_path, output_csv_path)
            
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

if __name__ == "__main__":
    main()